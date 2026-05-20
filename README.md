# IBKR GenAI Trading Agent

A LangGraph-based autonomous paper trading agent connected to Interactive Brokers via the TWS API, using Claude as the reasoning engine.

## Architecture

```
LangGraph Agent Loop
┌────────────────────────────────────────────────┐
│  agent_node ──→ tool_node ──→ agent_node ──→ … │
│       │              │                         │
│   Claude LLM    IBKR Tools                     │
│   (synthesis)   (data + execution)             │
└────────────────────────────────────────────────┘
         ↕                    ↕
    Anthropic API        IB Gateway / TWS
                         (paper trading)
```

The LLM synthesizes and makes qualitative judgments. All math (indicators, sizing, P&L) and all risk enforcement happen deterministically in Python.

## Setup

### 1. IBKR Gateway

Download [IB Gateway](https://www.interactivebrokers.com/en/trading/ib-api.php) and configure:

- **API Settings → Enable ActiveX and Socket Clients**: checked
- **Socket port**: 4002 (paper) or 7497 (TWS paper)
- **Trusted IPs**: 127.0.0.1
- **Read-Only API**: unchecked

### 2. Python Environment

```bash
cd ibkr_agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configuration

```bash
cp .env.example .env
# Edit .env with your Anthropic API key and IBKR settings
```

## Usage

### One-shot directive
```bash
python -m ibkr_agent run "Analyze AAPL and NVDA, take a position if warranted"
```

### Interactive mode
```bash
python -m ibkr_agent interactive
```

### Scheduled execution (market hours)
```bash
python -m ibkr_agent schedule
```

## Tools Available to the Agent

| Tool | Purpose |
|------|---------|
| `get_portfolio_snapshot` | Account state + all positions with P&L |
| `get_technical_summary` | Pre-computed indicator suite for any symbol |
| `place_trade` | Market or limit order with risk validation |
| `place_bracket_trade` | Atomic entry + take-profit + stop-loss |
| `close_position` | Fully exit an existing position |
| `get_news_sentiment` | Headlines + sentiment from IBKR/Polygon |

## Risk Guardrails

Enforced programmatically — the LLM cannot override:

- Max 5% of NLV in any single position
- Max 60% total portfolio exposure
- Min $100 order notional
- Max 20 trades per day
- Equities only (STK), USD only
- DAY orders only (no GTC for automated agent)

## Project Structure

```
ibkr_agent/
├── __init__.py          # Package metadata
├── __main__.py          # CLI entry point
├── config.py            # Risk limits, connection config, env loading
├── connection.py        # IBKR connection manager (singleton)
├── agent.py             # LangGraph graph: system prompt, nodes, edges
├── scheduler.py         # APScheduler market-hours job runner
├── audit.py             # Structured JSONL audit logging
└── tools/
    ├── __init__.py      # Tool exports
    ├── _helpers.py      # Shared IBKR utilities
    ├── portfolio.py     # Portfolio snapshot tool
    ├── technicals.py    # Technical analysis tool
    ├── execution.py     # Trade execution + risk guardrails
    └── news.py          # News/sentiment tool
```

## Logs

All agent activity is logged to `logs/`:

- `trades.jsonl` — Every order submission and rejection
- `agent.jsonl` — LLM decisions, tool calls, run metadata
- `agent_debug.log` — Full debug output

## Disclaimer

This is a paper trading system for simulation and research only. Nothing here constitutes investment advice.
