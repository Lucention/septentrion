import structlog

from septentrion.logging_setup import setup_logging


def test_setup_logging_binds_context_and_returns_logger():
    log = setup_logging(run_id="run-1", git_commit="abc123")
    assert log is not None
    # bound context vars are present on new loggers after setup
    ctx = structlog.contextvars.get_contextvars()
    assert ctx.get("run_id") == "run-1"
    assert ctx.get("git_commit") == "abc123"
