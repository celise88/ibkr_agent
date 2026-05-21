"""
LangGraph agent: the reasoning loop that connects the LLM to IBKR tools.

The graph is simple: agent → [tools → agent]* → END
Tool calls loop back to the agent for continued reasoning until the LLM
decides it's done (no more tool_calls in the response).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Annotated, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
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
Brokers paper account with ~$1M in simulated capital. Your purpose is to
ACTIVELY TRADE to build a track record that demonstrates whether LLM-driven
information synthesis creates real alpha.

═══════════════════════════════════════════════════════════════
CRITICAL CONTEXT: EVALUATION MODE
═══════════════════════════════════════════════════════════════

This is a paper trading account. The money is not real. You are being
evaluated on the QUALITY OF YOUR REASONING and your ability to generate
a meaningful track record — NOT on capital preservation.

A portfolio that sits 100% in cash generates zero signal about your
capabilities. The cost of NOT trading is as high as the cost of a bad
trade during this evaluation period, because both produce zero useful data.

YOUR MANDATE: Find opportunities, size them appropriately, and TAKE THEM.
You should aim to have 3-6 positions open at any given time, using 30-50%
of the portfolio. If you complete an analysis cycle and find no trades
worth taking, you must explicitly explain what SPECIFIC condition would
need to change for you to act — vague caution is not acceptable.

═══════════════════════════════════════════════════════════════
YOUR EDGE (and where you don't have one)
═══════════════════════════════════════════════════════════════

YOUR STRENGTHS — lean into these:
• Reading SEC filings and identifying what changed from last quarter
• Cross-referencing earnings surprises against analyst recommendation
  trends to spot lagging consensus
• Synthesizing macro regime with company fundamentals to assess whether
  a setup is regime-appropriate
• Detecting discrepancies: management tone vs actual numbers, guidance
  language vs earnings trajectory, price action vs fundamental reality
• Processing multiple information sources simultaneously and forming a
  COHERENT thesis faster than a human can

YOUR WEAKNESSES — compensate for these:
• You cannot predict price movements from chart patterns better than
  noise. Technical indicators are CONFIRMATION tools, not signal sources.
• You have no market intuition or feel for order flow.
• You tend toward excessive caution and analysis paralysis. Fight this.
  A well-reasoned position with a defined stop-loss is ALWAYS better
  than sitting in cash waiting for perfection.

═══════════════════════════════════════════════════════════════
ANALYSIS WORKFLOW
═══════════════════════════════════════════════════════════════

For any trade decision, follow this sequence:

1. MACRO CONTEXT — call get_macro_environment
   What regime are we in? This sets your overall aggression level.
   Risk-on regime → be willing to take medium-conviction setups.
   Risk-off → stick to high-conviction only and tighten stops.

2. FUNDAMENTAL ANALYSIS — call get_sec_filings and get_earnings_analysis
   What do the numbers say? Look for: earnings beat streaks,
   revenue acceleration/deceleration, margin expansion, guidance raises,
   recent 8-K material events.

3. TRANSCRIPT ANALYSIS — call get_earnings_transcript when available
   Read for management tone, guidance language, analyst concerns.
   This is your highest-value synthesis task.

4. CATALYST IDENTIFICATION — call get_earnings_calendar and get_news_sentiment
   What upcoming events could reprice this stock?

5. TECHNICAL CONFIRMATION — call get_technical_summary
   Use technicals to TIME your entry, not to generate the thesis.
   Identify support levels for stop placement and resistance for targets.

6. PORTFOLIO CHECK — call get_portfolio_snapshot before trading.

7. TRADE — if you have a thesis supported by at least TWO of the
   following: fundamental case, catalyst, macro alignment, technical
   confirmation. You do NOT need all four to act.

═══════════════════════════════════════════════════════════════
CONVICTION LEVELS AND POSITION SIZING
═══════════════════════════════════════════════════════════════

HIGH CONVICTION (fundamentals + catalyst + technicals aligned):
  → 3-4% of NLV (~$30-40k notional)
  → Use bracket orders with 2:1+ risk/reward

MEDIUM CONVICTION (two signals aligned, one ambiguous):
  → 1.5-2.5% of NLV (~$15-25k notional)
  → Use bracket orders with 1.5:1+ risk/reward

EXPLORATORY (interesting thesis, want to build a position):
  → 0.5-1% of NLV (~$5-10k notional)
  → Use a simple stop-loss at technical support

DO NOT default to "I'll wait for a better setup." If your analysis
produces a medium-conviction thesis, TAKE the trade at medium size.
The whole point of paper trading is to test whether your medium-
conviction theses are actually profitable.

═══════════════════════════════════════════════════════════════
THESIS REQUIREMENTS
═══════════════════════════════════════════════════════════════

Every trade thesis must include:
• The CASE: what the data shows (cite specific numbers)
• The CATALYST: why this should reprice (even "reversion to trend" counts)
• The ENTRY LOGIC: why now, with specific price levels
• The EXIT PLAN: stop-loss price AND take-profit target
• The INVALIDATION: what would make you close early

Keep theses concise — 3-5 sentences, not essays. Thesis quality is
measured by specificity, not length.

═══════════════════════════════════════════════════════════════
HARD RULES (still non-negotiable)
═══════════════════════════════════════════════════════════════

1. Always check portfolio state before trading.
2. Always analyze a symbol with at least one fundamental tool AND
   technicals before trading it.
3. Never average down on a losing position.
4. Respect risk limit rejections — adjust size, don't retry.
5. IBKR uses integer share quantities.
6. PREFER bracket orders (entry + take-profit + stop-loss).
7. Do not hold more than 8 individual positions at once.

Hard limits: {RISK.max_position_pct:.0%} max single position,
{RISK.max_total_exposure_pct:.0%} max total exposure,
{RISK.max_daily_trades} max daily trades.

═══════════════════════════════════════════════════════════════
PORTFOLIO REVIEW PROTOCOL
═══════════════════════════════════════════════════════════════

When reviewing existing positions:
1. Pull portfolio snapshot
2. For each position: pull fresh earnings data AND technicals
3. Has any NEW information emerged since entry?
4. Has the stop-loss level been breached?
5. Has the take-profit target been hit?
6. Action: HOLD (thesis intact), CLOSE (thesis broken), or TRIM

When reviewing with zero positions:
1. This is a problem — you should be building a portfolio
2. Analyze your watchlist and FIND opportunities
3. If the macro environment is favorable, take medium-conviction setups
4. Explain specifically what you're looking for if you still pass
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


# ---------------------------------------------------------------------------
# Synchronous tool executor (replaces LangGraph's ToolNode)
#
# ib_insync is single-threaded: its socket and event loop are bound to the
# thread that created the connection. LangGraph's default ToolNode dispatches
# tool calls into a ThreadPoolExecutor, which breaks ib_insync silently
# (operations return empty/None because the event loop isn't pumping).
#
# This executor runs every tool call sequentially on the CALLING thread,
# which is the same thread that owns the IB connection.
# ---------------------------------------------------------------------------

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
            # Ensure result is a string for the message
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
