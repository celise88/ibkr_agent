"""
Scheduled agent execution.

Runs the agent at configurable market-hours intervals using APScheduler.
Each job is a self-contained directive — the agent gets fresh portfolio
state on every run.

Usage:
    python -m ibkr_agent.scheduler
    # or import and customize:
    from ibkr_agent.scheduler import create_scheduler, add_custom_job
"""

from __future__ import annotations

import logging
import signal
import sys
from datetime import datetime, timezone

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
from apscheduler.schedulers.blocking import BlockingScheduler

from ibkr_agent.agent import run_agent
from ibkr_agent.audit import log_agent, setup_logging
from ibkr_agent.connection import disconnect

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job definitions
# ---------------------------------------------------------------------------

def morning_scan():
    """
    Run 5 minutes after market open.
    Check overnight gaps, review positions, scan watchlist.
    """
    logger.info("=== MORNING SCAN ===")
    try:
        run_agent(
            "Morning scan. Start with macro context — check the macro environment "
            "to understand today's regime. Then review our portfolio for any "
            "positions affected by overnight news or filings. Check the earnings "
            "calendar for any companies reporting today or this week. For any "
            "companies on our watchlist (AAPL, MSFT, NVDA, GOOGL, AMZN, META), "
            "pull earnings analysis and recent SEC filings. Only consider new "
            "positions if you find a fundamental catalyst the market hasn't "
            "fully priced — confirm with technicals before entering."
        )
    except Exception as exc:
        logger.error("Morning scan failed: %s", exc, exc_info=True)


def midday_review():
    """
    Midday portfolio health check.
    Review positions, check if theses still hold.
    """
    logger.info("=== MIDDAY REVIEW ===")
    try:
        run_agent(
            "Midday portfolio review. For each open position: (1) Check for any new "
            "8-K filings or material events since our last review, (2) Pull fresh "
            "earnings data to verify the fundamental thesis still holds, (3) Check "
            "technicals for any breakdown of key support/resistance levels. Also "
            "scan recent news for any macro developments (Fed commentary, economic "
            "data releases, geopolitical events) that change the portfolio's risk "
            "profile. Close any position where the original fundamental thesis "
            "has been invalidated by new information."
        )
    except Exception as exc:
        logger.error("Midday review failed: %s", exc, exc_info=True)


def afternoon_management():
    """
    Pre-close position management.
    Trim or close positions ahead of market close.
    """
    logger.info("=== AFTERNOON MANAGEMENT ===")
    try:
        run_agent(
            "Pre-close management. Review all open positions. For any position "
            "that is marginally profitable but showing weakening momentum, "
            "consider closing to lock in gains. For any intraday entries, "
            "decide whether to hold overnight or close. Do not open new "
            "positions this close to market close."
        )
    except Exception as exc:
        logger.error("Afternoon management failed: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------

def _job_listener(event):
    """Log job execution results."""
    if event.exception:
        log_agent("scheduled_job_error", {
            "job_id": event.job_id,
            "error": str(event.exception),
        })
    else:
        log_agent("scheduled_job_success", {
            "job_id": event.job_id,
            "retval_type": type(event.retval).__name__ if event.retval else None,
        })


def create_scheduler() -> BlockingScheduler:
    """
    Create and configure the scheduler with default market-hours jobs.

    Returns a BlockingScheduler — call .start() to begin the run loop.
    Jobs run Monday-Friday only, using US Eastern times.
    """
    scheduler = BlockingScheduler(timezone="US/Eastern")
    scheduler.add_listener(_job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

    # 9:35 AM ET — 5 minutes after market open
    scheduler.add_job(
        morning_scan,
        "cron",
        id="morning_scan",
        day_of_week="mon-fri",
        hour=9,
        minute=35,
        misfire_grace_time=300,
    )

    # 12:00 PM ET — midday review
    scheduler.add_job(
        midday_review,
        "cron",
        id="midday_review",
        day_of_week="mon-fri",
        hour=12,
        minute=0,
        misfire_grace_time=300,
    )

    # 3:45 PM ET — 15 minutes before close
    scheduler.add_job(
        afternoon_management,
        "cron",
        id="afternoon_management",
        day_of_week="mon-fri",
        hour=15,
        minute=45,
        misfire_grace_time=300,
    )

    logger.info(
        "Scheduler configured with %d jobs: %s",
        len(scheduler.get_jobs()),
        [j.id for j in scheduler.get_jobs()],
    )

    return scheduler


def add_custom_job(
    scheduler: BlockingScheduler,
    job_id: str,
    directive: str,
    hour: int,
    minute: int = 0,
    day_of_week: str = "mon-fri",
) -> None:
    """
    Add a custom scheduled directive to an existing scheduler.

    Example:
        add_custom_job(scheduler, "sector_scan", "Analyze XLK, XLF, XLE...", hour=10, minute=30)
    """
    def _custom_job():
        logger.info("=== CUSTOM JOB: %s ===", job_id)
        try:
            run_agent(directive)
        except Exception as exc:
            logger.error("Custom job '%s' failed: %s", job_id, exc, exc_info=True)

    scheduler.add_job(
        _custom_job,
        "cron",
        id=job_id,
        day_of_week=day_of_week,
        hour=hour,
        minute=minute,
        misfire_grace_time=300,
    )
    logger.info("Added custom job: %s at %02d:%02d %s", job_id, hour, minute, day_of_week)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    setup_logging()
    logger.info("Starting IBKR Trading Agent Scheduler")
    logger.info("Press Ctrl+C to stop.")

    scheduler = create_scheduler()

    # Graceful shutdown on SIGINT/SIGTERM
    def shutdown(signum, frame):
        logger.info("Shutdown signal received. Cleaning up...")
        scheduler.shutdown(wait=False)
        disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
    finally:
        disconnect()


if __name__ == "__main__":
    main()
