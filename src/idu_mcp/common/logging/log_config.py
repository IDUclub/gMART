import sys
from pathlib import Path

from loguru import logger

from src.idu_mcp.common.logging.log_rotation import start_daily_log_rotation

_LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}"
)


def _get_project_root() -> Path:
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current_path.parents[4]


def config_logger(log_file_name: str = "idu_mcp.log", level: str = "INFO") -> Path:
    """
    Configure idu_mcp logging to stderr and a file in the project root.

    The file is rotated by size once a day (see ``log_rotation``).

    Args:
        log_file_name (str): Name of the log file created in the project root.
        level (str): Minimum log level for both sinks.
    Returns:
        Path: Absolute path to the log file.
    """

    logger.info("Starting logger initialization")
    log_path = _get_project_root() / log_file_name
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=_LOG_FORMAT,
    )
    logger.add(
        log_path,
        level=level,
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=False,
        format=_LOG_FORMAT,
    )
    start_daily_log_rotation(log_path)
    logger.info(f"Configured logger, writing logs to {log_path}")
    return log_path
