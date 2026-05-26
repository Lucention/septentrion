"""structlog configuration: human console by default, context bound per run."""

from __future__ import annotations

import logging

import structlog


def setup_logging(
    run_id: str, git_commit: str, level: int = logging.INFO
) -> structlog.stdlib.BoundLogger:
    """Configure structlog and bind run-wide provenance context."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(run_id=run_id, git_commit=git_commit)
    return structlog.get_logger()
