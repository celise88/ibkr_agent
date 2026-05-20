"""
CLI entry point for the IBKR GenAI Trading Agent.

Modes:
    python -m ibkr_agent run "Analyze AAPL and take a position if warranted"
    python -m ibkr_agent interactive
    python -m ibkr_agent schedule
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from ibkr_agent.agent import run_agent
from ibkr_agent.audit import setup_logging
from ibkr_agent.connection import disconnect, get_connection


def _print_messages(messages):
    """Pretty-print the agent's message history."""
    for msg in messages:
        role = msg.type.upper()
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        if role == "SYSTEM":
            continue  # Don't dump the system prompt
        elif role == "TOOL":
            # Truncate long tool outputs
            name = getattr(msg, "name", "tool")
            if len(content) > 600:
                content = content[:600] + f"\n  ... ({len(content)} chars total)"
            print(f"\n  [{name}] {content}")
        elif role == "AI":
            # Show tool calls if present
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    args_str = ", ".join(
                        f"{k}={repr(v)[:50]}" for k, v in tc.get("args", {}).items()
                    )
                    print(f"\n  → {tc['name']}({args_str})")
            if content:
                print(f"\n{'─' * 60}")
                print(f"AGENT:\n{content}")
                print(f"{'─' * 60}")
        elif role == "HUMAN":
            print(f"\nYOU: {content}")


def cmd_run(args):
    """Execute a single directive and exit."""
    directive = " ".join(args.directive)
    if not directive.strip():
        print("Error: provide a directive. Example:")
        print('  python -m ibkr_agent run "Analyze AAPL and NVDA"')
        sys.exit(1)

    print(f"Directive: {directive}\n")
    try:
        messages = run_agent(directive)
        _print_messages(messages)
    finally:
        disconnect()


def cmd_interactive(args):
    """Interactive REPL — type directives, see agent responses."""
    print("IBKR Trading Agent — Interactive Mode")
    print("Type a directive (or 'quit' to exit, 'help' for examples).\n")

    examples = textwrap.dedent("""\
        Example directives:
          • Check our portfolio and summarize position health
          • Analyze AAPL, MSFT, NVDA — rank by setup quality
          • If AAPL shows a bullish setup, take a small position
          • Review all positions and close anything with a broken thesis
          • Close our AAPL position — momentum has reversed
          • Scan AMD, AVGO, MRVL for semiconductor sector setups
    """)

    try:
        # Verify connection on startup
        get_connection()
        print("Connected to IBKR.\n")

        while True:
            try:
                directive = input("▶ ").strip()
            except EOFError:
                break

            if not directive:
                continue
            if directive.lower() in {"quit", "exit", "q"}:
                break
            if directive.lower() == "help":
                print(examples)
                continue

            try:
                messages = run_agent(directive)
                _print_messages(messages)
            except Exception as exc:
                print(f"\nError during agent execution: {exc}")
                print("The agent encountered an error. You can try again or type 'quit'.\n")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        disconnect()
        print("Disconnected. Goodbye.")


def cmd_schedule(args):
    """Start the scheduled execution loop."""
    from ibkr_agent.scheduler import main as scheduler_main
    scheduler_main()


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        prog="ibkr_agent",
        description="GenAI Trading Agent for IBKR Paper Trading",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run
    run_parser = subparsers.add_parser(
        "run", help="Execute a single trading directive"
    )
    run_parser.add_argument(
        "directive", nargs="+", help="Natural-language trading directive"
    )
    run_parser.set_defaults(func=cmd_run)

    # interactive
    interactive_parser = subparsers.add_parser(
        "interactive", help="Start interactive REPL mode"
    )
    interactive_parser.set_defaults(func=cmd_interactive)

    # schedule
    schedule_parser = subparsers.add_parser(
        "schedule", help="Start scheduled market-hours execution"
    )
    schedule_parser.set_defaults(func=cmd_schedule)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
