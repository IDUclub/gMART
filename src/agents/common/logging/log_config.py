import logging
import sys
from pathlib import Path

from loguru import logger

from src.agents.common.logging.log_rotation import start_daily_log_rotation


def _get_project_root() -> Path:
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current_path.parents[4]


class _InterceptHandler(logging.Handler):
    """
    Forward standard-library ``logging`` records into loguru.

    The app configures only loguru sinks, but uvicorn, starlette, asyncio and
    most third-party libraries log through stdlib ``logging``. Without this
    bridge their records never reach the loguru file sink, so failures that
    surface through them look like the stream simply stopped with nothing in the
    logs.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _intercept_stdlib_logging(level: str) -> None:
    """
    Route stdlib ``logging`` (starlette, asyncio, third-party libs) into loguru.

    Any logger that propagates to the root logger — i.e. everything that does not
    install its own handlers, like uvicorn does — will be captured by the loguru
    sinks configured in :func:`config_logger`.

    Args:
        level (str): Minimum level for the intercepted root logger.
    """

    logging.basicConfig(handlers=[_InterceptHandler()], level=level, force=True)


def config_logger(log_file_name: str = ".log", level: str = "INFO") -> Path:
    """
    Configure application logging to a file in the project root.

    The file is rotated by size once a day (see ``log_rotation``).

    Returns:
        Path: Absolute path to the log file.
    """

    logger.info("Starting logger initialization")
    log_path = _get_project_root() / log_file_name
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}",
    )
    logger.add(
        log_path,
        level=level,
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}",
    )
    _intercept_stdlib_logging(level)
    start_daily_log_rotation(log_path)
    logger.info("Configured logger")
    return log_path
