"""
Two-phase trading agent: Analyst → Trader.

Phase 1 (Analyst): Researches using all information tools. Produces a
structured analysis with trade recommendations. This phase works well
already — the agent does thorough research.

Phase 2 (Trader): Receives the analyst's output and MUST act on it.
The trader has access ONLY to execution tools plus a `decline_trade`
tool. The forcing function: ending a turn without calling ANY tool is
not possible — the trader must either execute or explicitly decline
with structured justification.

This architecture solves the "analysis paralysis" problem where the
agent researches thoroughly but never pulls the trigger.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import time
from typing import Annotated, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph_sdk.errors import BadRequestError

from ibkr_agent.audit import log_agent, log_trade
from ibkr_agent.config import GROQ_API_KEY, GROQ_MODEL, LLM, RISK
from ibkr_agent.tools import ALL_TOOLS
from ibkr_agent.tools.portfolio import get_portfolio_snapshot
from ibkr_agent.tools.technicals import get_technical_summary
from ibkr_agent.tools.execution import place_trade, place_bracket_trade, close_position

logger = logging.getLogger(__name__)


def _load_groq_llm() -> Any | None:
    if not GROQ_API_KEY:
        return None

    for module_name in ("langchain_groq", "langchain_groq.chat_models"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue

        ChatGroq = getattr(module, "ChatGroq", None)
        if ChatGroq is None:
            continue

        try:
            return ChatGroq(
                model=GROQ_MODEL,
                api_key=GROQ_API_KEY,
                temperature=LLM.temperature,
                max_tokens=LLM.max_tokens,
            )
        except TypeError:
            return ChatGroq(
                model=GROQ_MODEL,
                api_key=GROQ_API_KEY,
                temperature=LLM.temperature,
            )

    return None


def _invoke_with_groq_fallback(llm: Any, messages: list[BaseMessage], phase_name: str) -> Any:
    try:
        return llm.invoke(messages)
    except BadRequestError as exc:
        if exc.status_code != 400:
            raise
        groq_llm = _load_groq_llm()
        if groq_llm is None:
            logger.error(
                "Anthropic returned HTTP 400 but GROQ fallback is unavailable. "
                "Set GROQ_API_KEY and install langchain_groq."
            )
            raise
        logger.warning("Anthropic returned HTTP 400; falling back to GROQ for %s", phase_name)
        return groq_llm.invoke(messages)
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        if status_code == 400:
            groq_llm = _load_groq_llm()
            if groq_llm is not None:
                logger.warning("Anthropic returned HTTP 400; falling back to GROQ for %s", phase_name)
                return groq_llm.invoke(messages)
        raise


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgentState(dict):
    messages: Annotated[list[BaseMessage], add_messages]
    daily_trade_count: int


# ---------------------------------------------------------------------------
# The decline_trade tool — makes NOT trading an active, auditable choice
# ---------------------------------------------------------------------------

from langchain_core.tools import tool


@tool
def decline_trade(
    symbol: str,
    reason: str,
    what_would_change_your_mind: str,
    reentry_price: float | None = None,
    reassess_in_hours: int = 24,
) -> dict:
    """
    Explicitly decline to trade a symbol. You MUST call this for every
    symbol you analyzed but chose NOT to trade. This is mandatory —
    you cannot end your turn without either trading or declining each
    symbol in your analysis.

    Args:
        symbol: The ticker you're passing on.
        reason: Specific, data-grounded reason (not "waiting for a better setup").
        what_would_change_your_mind: What condition would make this tradeable.
        reentry_price: Price level that would trigger reconsideration.
        reassess_in_hours: When to look at this again (default 24).

    Returns:
        Confirmation that the decline was logged.
    """
    result = {
        "status": "DECLINED",
        "symbol": symbol.upper(),
        "reason": reason,
        "trigger_condition": what_would_change_your_mind,
        "reentry_price": reentry_price,
        "reassess_in_hours": reassess_in_hours,
    }

    log_trade("trade_declined", result)
    logger.info(
        "Trade declined: %s — %s (reconsider at $%s or in %dh)",
        symbol, reason[:80],
        reentry_price or "N/A",
        reassess_in_hours,
    )

    return result


# ---------------------------------------------------------------------------
# Tool sets for each phase
# ---------------------------------------------------------------------------

# Phase 1: all information + execution tools
ANALYST_TOOLS = ALL_TOOLS

# Phase 2: execution only — the trader can check portfolio/technicals
# for final confirmation but primarily must execute or decline
TRADER_TOOLS = [
    get_portfolio_snapshot,
    get_technical_summary,
    place_trade,
    place_bracket_trade,
    close_position,
    decline_trade,
]

# Name→tool lookups for sync execution
_ANALYST_TOOL_MAP: dict[str, Any] = {t.name: t for t in ANALYST_TOOLS}
_TRADER_TOOL_MAP: dict[str, Any] = {t.name: t for t in TRADER_TOOLS}


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

ANALYST_PROMPT = f"""\
You are the ANALYST phase of a two-phase trading system connected to an
Interactive Brokers paper account with ~$1M simulated capital.

YOUR JOB: Research and produce SPECIFIC TRADE RECOMMENDATIONS. You do NOT
execute trades — a separate Trader agent will act on your recommendations.

Your output will be handed directly to the Trader, so be specific:
- Name the symbols you recommend trading
- Specify BUY or SELL
- Suggest position size (% of NLV and approximate share count)
- Provide entry price, stop-loss price, and take-profit target
- State your conviction level (high/medium/exploratory)
- Give a concise thesis (3-5 sentences with specific data points)

WORKFLOW:
1. Check macro environment for regime context
2. Check portfolio for current positions
3. For each symbol on the watchlist or earnings calendar:
   - Pull earnings analysis and SEC filings
   - Pull earnings transcript if available
   - Pull technicals for entry/exit levels
4. Produce a structured recommendation for EACH symbol analyzed

YOU MUST RECOMMEND AT LEAST ONE TRADE per analysis session. If you
genuinely cannot find a single opportunity across 6+ major stocks in a
favorable macro environment, your standards are too high. A paper trading
evaluation with 100% cash allocation is a failure state.

Target portfolio: 3-6 positions, 30-50% deployed.

Conviction sizing:
- High: 3-4% NLV (~$30-40k)
- Medium: 1.5-2.5% NLV (~$15-25k)
- Exploratory: 0.5-1% NLV (~$5-10k)

Hard limits: {RISK.max_position_pct:.0%} max single position,
{RISK.max_total_exposure_pct:.0%} max total exposure.
"""

TRADER_PROMPT = """\
You are the TRADER phase of a two-phase trading system. The Analyst has
completed research and provided trade recommendations below.

YOUR JOB: Execute the trades or explicitly decline each one.

RULES:
1. For EVERY symbol the Analyst recommended, you MUST either:
   a) Call place_bracket_trade or place_trade to execute, OR
   b) Call decline_trade with a specific, data-grounded reason

2. You CANNOT end your turn without addressing every recommendation.
   Ending your turn without calling tools is a system error.

3. Before your first trade, call get_portfolio_snapshot to confirm
   current state.

4. You may call get_technical_summary for final price confirmation
   before setting bracket levels.

5. For bracket orders, use the Analyst's suggested stop and target
   levels, adjusted for current price if it has moved.

6. DO NOT re-analyze or second-guess the Analyst's fundamental thesis.
   Your job is EXECUTION and RISK MANAGEMENT, not re-doing the research.
   The only valid reasons to decline are:
   - Price has moved significantly since analysis (>2% from suggested entry)
   - Risk limits would be violated
   - The position would create unwanted concentration

7. After executing all trades/declines, provide a brief summary of
   what was done.

THE ANALYST'S RECOMMENDATIONS:
"""

REVIEW_PROMPT = f"""\
You are a portfolio management agent connected to an Interactive Brokers
paper account. Review all current positions and take action.

For EACH open position, you MUST either:
  a) HOLD — state why the thesis is intact (cite current data)
  b) CLOSE — call close_position with a reason
  c) TRIM — call place_trade to reduce the position

You also MUST call decline_trade for any position you choose to HOLD,
documenting your hold thesis and what would trigger an exit.

If the portfolio has fewer than 3 positions AND gross exposure is below
30%, flag this and recommend symbols to analyze at the next morning scan.

Hard limits: {RISK.max_position_pct:.0%} max single position,
{RISK.max_total_exposure_pct:.0%} max total exposure.
"""


# ---------------------------------------------------------------------------
# LLM instances
# ---------------------------------------------------------------------------

llm = ChatAnthropic(
    model=LLM.model,
    temperature=LLM.temperature,
    max_tokens=LLM.max_tokens,
)

analyst_llm = llm.bind_tools(ANALYST_TOOLS)
trader_llm = llm.bind_tools(TRADER_TOOLS)


# ---------------------------------------------------------------------------
# Sync tool executors
# ---------------------------------------------------------------------------

def _make_sync_tool_node(tool_map: dict[str, Any]):
    """Factory for synchronous tool executor nodes."""

    def sync_tool_node(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])

        tool_messages = []
        for tc in tool_calls:
            tool_name = tc["name"]
            tool_args = tc.get("args", {})
            tool_id = tc.get("id", tool_name)

            tool_fn = tool_map.get(tool_name)
            if tool_fn is None:
                tool_messages.append(ToolMessage(
                    content=f"Error: unknown tool '{tool_name}'. Available: {list(tool_map.keys())}",
                    tool_call_id=tool_id,
                    name=tool_name,
                ))
                continue

            try:
                result = tool_fn.invoke(tool_args)
                if not isinstance(result, str):
                    result = json.dumps(result, default=str, indent=2)
                tool_messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_id,
                    name=tool_name,
                ))
            except Exception as exc:
                logger.error("Tool '%s' raised: %s", tool_name, exc, exc_info=True)
                tool_messages.append(ToolMessage(
                    content=f"Error executing {tool_name}: {exc}",
                    tool_call_id=tool_id,
                    name=tool_name,
                ))

        return {"messages": tool_messages}

    return sync_tool_node


analyst_tool_node = _make_sync_tool_node(_ANALYST_TOOL_MAP)
trader_tool_node = _make_sync_tool_node(_TRADER_TOOL_MAP)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def _make_agent_node(bound_llm, prompt: str, phase_name: str):
    """Factory for agent reasoning nodes."""

    def agent_node(state: AgentState) -> dict:
        messages = state.get("messages", [])

        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=prompt)] + messages

        t0 = time.monotonic()
        response = _invoke_with_groq_fallback(bound_llm, messages, phase_name)
        elapsed = time.monotonic() - t0

        tool_calls = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls = [
                {"name": tc["name"], "args_summary": _summarize_args(tc.get("args", {}))}
                for tc in response.tool_calls
            ]

        log_agent(f"{phase_name}_response", {
            "elapsed_sec": round(elapsed, 2),
            "tool_calls": tool_calls,
            "content_length": len(response.content) if isinstance(response.content, str) else 0,
            "stop_reason": getattr(response, "response_metadata", {}).get("stop_reason"),
        })

        return {"messages": [response]}

    return agent_node


def _summarize_args(args: dict) -> dict:
    return {
        k: v if isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) < 200)
        else f"<{type(v).__name__}:{len(str(v))}chars>"
        for k, v in args.items()
    }


def route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Build separate graphs for each phase
# ---------------------------------------------------------------------------

def _build_phase_graph(agent_node_fn, tool_node_fn):
    """Build a standard agent→tools loop graph."""
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node_fn)
    workflow.add_node("tools", tool_node_fn)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", END: END}
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()


analyst_graph = _build_phase_graph(
    _make_agent_node(analyst_llm, ANALYST_PROMPT, "analyst"),
    analyst_tool_node,
)

trader_graph = _build_phase_graph(
    _make_agent_node(trader_llm, TRADER_PROMPT, "trader"),
    trader_tool_node,
)

review_graph = _build_phase_graph(
    _make_agent_node(trader_llm, REVIEW_PROMPT, "reviewer"),
    trader_tool_node,
)


# ---------------------------------------------------------------------------
# Two-phase runner
# ---------------------------------------------------------------------------

def _extract_analysis(messages: list[BaseMessage]) -> str:
    """Extract the analyst's final text output from the message history."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            return msg.content
    return "No analysis produced — the analyst phase did not generate recommendations."


def run_agent(directive: str) -> list[BaseMessage]:
    """
    Execute the two-phase analyst→trader pipeline.

    Phase 1: Analyst researches and produces trade recommendations.
    Phase 2: Trader executes each recommendation or explicitly declines.

    Returns the combined message history from both phases.
    """
    logger.info("═══ PHASE 1: ANALYST ═══")
    logger.info("Directive: %s", directive[:200])
    t0 = time.monotonic()

    # --- Phase 1: Analyst ---
    analyst_result = analyst_graph.invoke({
        "messages": [HumanMessage(content=directive)],
        "daily_trade_count": 0,
    })

    analyst_messages = analyst_result["messages"]
    analysis_text = _extract_analysis(analyst_messages)

    analyst_elapsed = time.monotonic() - t0
    analyst_tool_count = sum(
        len(m.tool_calls) for m in analyst_messages
        if hasattr(m, "tool_calls") and m.tool_calls
    )

    log_agent("analyst_complete", {
        "directive": directive[:500],
        "elapsed_sec": round(analyst_elapsed, 2),
        "message_count": len(analyst_messages),
        "tool_call_count": analyst_tool_count,
        "analysis_length": len(analysis_text),
    })

    logger.info(
        "Analyst complete: %d messages, %d tool calls, %.0fs. Analysis: %d chars.",
        len(analyst_messages), analyst_tool_count, analyst_elapsed, len(analysis_text),
    )

    # --- Phase 2: Trader ---
    logger.info("═══ PHASE 2: TRADER ═══")
    logger.info("Feeding analysis to trader (%d chars)...", len(analysis_text))

    trader_directive = (
        f"The Analyst has completed research. Here are the recommendations:\n\n"
        f"---\n{analysis_text}\n---\n\n"
        f"Execute each recommendation or decline with justification. "
        f"You MUST call a tool for every symbol mentioned above."
    )

    t1 = time.monotonic()

    trader_result = trader_graph.invoke({
        "messages": [HumanMessage(content=trader_directive)],
        "daily_trade_count": 0,
    })

    trader_messages = trader_result["messages"]
    trader_elapsed = time.monotonic() - t1
    trader_tool_count = sum(
        len(m.tool_calls) for m in trader_messages
        if hasattr(m, "tool_calls") and m.tool_calls
    )

    # Count actual trades vs declines
    trade_count = 0
    decline_count = 0
    for msg in trader_messages:
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content) if isinstance(msg.content, str) else {}
                status = data.get("status", "")
                if status == "SUBMITTED":
                    trade_count += 1
                elif status == "DECLINED":
                    decline_count += 1
                elif status == "REJECTED":
                    decline_count += 1
            except (json.JSONDecodeError, AttributeError):
                pass

    log_agent("trader_complete", {
        "elapsed_sec": round(trader_elapsed, 2),
        "message_count": len(trader_messages),
        "tool_call_count": trader_tool_count,
        "trades_executed": trade_count,
        "trades_declined": decline_count,
    })

    total_elapsed = time.monotonic() - t0
    logger.info(
        "Trader complete: %d trades executed, %d declined, %.0fs.",
        trade_count, decline_count, trader_elapsed,
    )
    logger.info(
        "═══ PIPELINE COMPLETE: %.0fs total, %d trades ═══",
        total_elapsed, trade_count,
    )

    log_agent("run_complete", {
        "directive": directive[:500],
        "elapsed_sec": round(total_elapsed, 2),
        "analyst_messages": len(analyst_messages),
        "trader_messages": len(trader_messages),
        "analyst_tool_calls": analyst_tool_count,
        "trader_tool_calls": trader_tool_count,
        "trades_executed": trade_count,
        "trades_declined": decline_count,
    })

    return analyst_messages + trader_messages


def run_review(directive: str) -> list[BaseMessage]:
    """
    Execute a portfolio review — uses the review graph which has
    execution tools and the decline_trade forcing function.
    """
    logger.info("═══ PORTFOLIO REVIEW ═══")
    logger.info("Directive: %s", directive[:200])
    t0 = time.monotonic()

    result = review_graph.invoke({
        "messages": [HumanMessage(content=directive)],
        "daily_trade_count": 0,
    })

    messages = result["messages"]
    elapsed = time.monotonic() - t0
    tool_count = sum(
        len(m.tool_calls) for m in messages
        if hasattr(m, "tool_calls") and m.tool_calls
    )

    log_agent("review_complete", {
        "directive": directive[:500],
        "elapsed_sec": round(elapsed, 2),
        "message_count": len(messages),
        "tool_call_count": tool_count,
    })

    logger.info("Review complete: %d messages, %d tool calls, %.0fs.", len(messages), tool_count, elapsed)

    return messages
