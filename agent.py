"""
LangGraph agent: the reasoning loop that connects the LLM to IBKR tools.

The graph is simple: agent → [tools → agent]* → END
Tool calls loop back to the agent for continued reasoning until the LLM
decides it's done (no more tool_calls in the response).
"""

from __future__ import annotations

import logging
import time
from typing import Annotated, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import json          # add this
import logging
import time
from typing import Annotated, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage  # add ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from ibkr_agent.audit import log_agent
from ibkr_agent.config import LLM, RISK
from ibkr_agent.tools import ALL_TOOLS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgentState(dict):
    """
    LangGraph state container.

    messages:           Full conversation history (LLM + tool results).
    daily_trade_count:  Running count of trades placed this session.
                        Enforced as a circuit breaker in the system prompt;
                        hard enforcement is in the execution tool.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    daily_trade_count: int


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""\
You are an information-synthesis trading agent connected to an Interactive
Brokers paper account. Your EDGE over human traders is NOT chart reading —
it is your ability to rapidly ingest, cross-reference, and synthesize large
volumes of unstructured information (SEC filings, earnings data, macro
indicators, news) into actionable trade theses.

═══════════════════════════════════════════════════════════════
WHERE YOU ADD VALUE (and where you don't)
═══════════════════════════════════════════════════════════════

YOUR STRENGTHS — lean into these:
• Reading a 50-page 10-K and identifying what changed from last year
• Cross-referencing earnings surprises against analyst recommendation
  trends to spot lagging consensus
• Synthesizing macro regime (rates, VIX, credit spreads) with
  company-level fundamentals to assess whether a setup is regime-
  appropriate
• Detecting discrepancies: management tone vs actual numbers,
  guidance language vs earnings trajectory, price action vs
  fundamental reality
• Processing multiple information sources simultaneously — filings,
  earnings, macro, technicals, news — and forming a COHERENT thesis

YOUR WEAKNESSES — compensate for these:
• You cannot predict price movements from chart patterns better than
  noise. Technical indicators are CONFIRMATION tools, not primary
  signal sources.
• You have no real-time market intuition or "feel" for order flow.
• You are prone to constructing plausible-sounding narratives that
  aren't backed by data. ALWAYS ground your thesis in specific
  numbers from your tools.

═══════════════════════════════════════════════════════════════
ANALYSIS WORKFLOW (information-first, not chart-first)
═══════════════════════════════════════════════════════════════

For any trade decision, follow this sequence:

1. MACRO CONTEXT — call get_macro_environment
   What regime are we in? Risk-on or risk-off? Rates rising or falling?
   This determines whether you should be aggressive or defensive.

2. FUNDAMENTAL ANALYSIS — call get_sec_filings and get_earnings_analysis
   What do the actual numbers say? Earnings trend? Revenue trajectory?
   Cash flow vs earnings divergence? Recent 8-K material events?
   Compare actual results to estimates — is the company consistently
   beating and by how much?

3. CATALYST IDENTIFICATION — call get_earnings_calendar and get_news_sentiment
   What's coming up? Pre-earnings is where information synthesis creates
   the most alpha. Is there a catalyst that the market hasn't priced?

4. TECHNICAL CONFIRMATION — call get_technical_summary
   Technicals are the LAST step, not the first. Use them to confirm
   timing and identify entry/exit levels for a fundamentally-grounded
   thesis. A bullish fundamental thesis with bearish technicals means
   WAIT, not SKIP.

5. PORTFOLIO CHECK — call get_portfolio_snapshot
   Before any trade, know your current state.

6. TRADE OR PASS — if and only if you have a thesis grounded in
   fundamentals AND confirmed by technicals AND appropriate for the
   macro regime.

═══════════════════════════════════════════════════════════════
THESIS REQUIREMENTS
═══════════════════════════════════════════════════════════════

Every trade thesis MUST include:
• The FUNDAMENTAL case: specific numbers from filings or earnings
• The CATALYST: what will cause the market to reprice this information
• The TECHNICAL entry: why NOW is the right time (not just "RSI is low")
• The MACRO context: why this trade fits the current environment
• The INVALIDATION: what specific condition would make you exit

A thesis that says "RSI is oversold and MACD is crossing up" is NOT
sufficient. A thesis that says "AAPL beat EPS estimates by 8% for the
4th consecutive quarter, guidance was raised, but the stock pulled back
3% on a broad market selloff — technicals show support at the 50-day
EMA, VIX is elevated but declining, and the yield curve is steepening
which favors growth names" IS sufficient.

═══════════════════════════════════════════════════════════════
HARD RULES
═══════════════════════════════════════════════════════════════

1. Never trade without checking portfolio state first.
2. Never trade a symbol you haven't analyzed with BOTH fundamental
   and technical tools in this session.
3. If your information sources conflict, DO NOTHING. Cash is a position.
4. Never average down on a losing position.
5. Respect all risk limit rejections.
6. IBKR uses integer share quantities.
7. Prefer bracket orders (entry + take-profit + stop-loss).

Position sizing:
- High conviction (fundamentals + technicals + macro aligned): ~3-4% NLV
- Medium conviction (2 of 3 aligned): ~1.5-2% NLV
- Exploratory: ~0.5-1% NLV

Hard limits: {RISK.max_position_pct:.0%} max single position,
{RISK.max_total_exposure_pct:.0%} max total exposure,
{RISK.max_daily_trades} max daily trades.

═══════════════════════════════════════════════════════════════
PORTFOLIO REVIEW PROTOCOL
═══════════════════════════════════════════════════════════════

When reviewing existing positions:
1. Pull portfolio snapshot
2. For each position: pull fresh earnings data AND technicals
3. Ask: does the original FUNDAMENTAL thesis still hold?
4. Ask: has any new information emerged (8-K filings, guidance changes)?
5. Recommend HOLD (thesis intact), CLOSE (thesis broken), or TRIM
"""


# ---------------------------------------------------------------------------
# LLM and tools
# ---------------------------------------------------------------------------

llm = ChatAnthropic(
    model=LLM.model,
    temperature=LLM.temperature,
    max_tokens=LLM.max_tokens,
)
llm_with_tools = llm.bind_tools(ALL_TOOLS)

# Build a name→tool lookup for the sync executor
_TOOL_MAP: dict[str, Any] = {t.name: t for t in ALL_TOOLS}


def sync_tool_node(state: AgentState) -> dict:
    """
    Execute tool calls from the last AI message synchronously.
    Returns ToolMessage results for each call.
    """
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    tool_messages = []
    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc.get("args", {})
        tool_id = tc.get("id", tool_name)

        tool_fn = _TOOL_MAP.get(tool_name)
        if tool_fn is None:
            tool_messages.append(ToolMessage(
                content=f"Error: unknown tool '{tool_name}'",
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


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def agent_node(state: AgentState) -> dict:
    """
    Core reasoning node. Prepends the system prompt if not already present,
    then invokes the LLM with tool bindings.
    """
    messages = state.get("messages", [])

    # Inject system prompt once
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    t0 = time.monotonic()
    response = llm_with_tools.invoke(messages)
    elapsed = time.monotonic() - t0

    # Audit: log what the LLM decided
    tool_calls = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_calls = [
            {"name": tc["name"], "args_summary": _summarize_args(tc.get("args", {}))}
            for tc in response.tool_calls
        ]

    log_agent("llm_response", {
        "elapsed_sec": round(elapsed, 2),
        "tool_calls": tool_calls,
        "content_length": len(response.content) if isinstance(response.content, str) else 0,
        "stop_reason": getattr(response, "response_metadata", {}).get("stop_reason"),
    })

    return {"messages": [response]}


def _summarize_args(args: dict) -> dict:
    """Truncate large args for logging (don't dump full tool results into the log)."""
    return {
        k: v if isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) < 200)
        else f"<{type(v).__name__}:{len(str(v))}chars>"
        for k, v in args.items()
    }


def route_after_agent(state: AgentState) -> str:
    """
    Conditional edge: if the LLM made tool calls, route to the tool node.
    Otherwise, the agent is done — route to END.
    """
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Construct and compile the agent graph.

    The graph is:
        agent ──[has tool calls?]──> tools ──> agent (loop)
               └──[no tool calls]──> END
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", sync_tool_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# Module-level compiled graph (import and use directly)
graph = build_graph()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_agent(directive: str) -> list[BaseMessage]:
    """
    Execute one full agent loop with a natural-language directive.

    Examples:
        run_agent("Analyze AAPL and NVDA. If either shows a clear setup, take a position.")
        run_agent("Review all open positions and close anything that's deteriorated.")
        run_agent("Scan NVDA, AMD, AVGO — rank by setup quality and take your top pick.")
        run_agent("Close our AAPL position — thesis has broken down.")

    Returns:
        The full message history (SystemMessage, HumanMessage, AIMessage,
        ToolMessage, ...) for inspection or display.
    """
    logger.info("Agent directive: %s", directive[:200])
    t0 = time.monotonic()

    result = graph.invoke({
        "messages": [HumanMessage(content=directive)],
        "daily_trade_count": 0,
    })

    elapsed = time.monotonic() - t0
    message_count = len(result["messages"])
    tool_call_count = sum(
        len(m.tool_calls) for m in result["messages"]
        if hasattr(m, "tool_calls") and m.tool_calls
    )

    logger.info(
        "Agent completed in %.1fs — %d messages, %d tool calls.",
        elapsed, message_count, tool_call_count,
    )

    log_agent("run_complete", {
        "directive": directive[:500],
        "elapsed_sec": round(elapsed, 2),
        "message_count": message_count,
        "tool_call_count": tool_call_count,
    })

    return result["messages"]
