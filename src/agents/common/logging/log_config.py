import sys
from pathlib import Path

from loguru import logger


def _get_project_root() -> Path:
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current_path.parents[4]


def config_logger(log_file_name: str = ".log", level: str = "INFO") -> Path:
    """
    Configure application logging to a file in the project root.

    Returns:
        Path: Absolute path to the log file.
    """

    logger.info("Starting logger initialization")
    log_path = _get_project_root() / log_file_name
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}",
    )
    logger.add(
        log_path,
        level=level,
        rotation="100 MB",
        retention="31 days",
        compression="zip",
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}",
    )
    logger.info("Configured logger")
    return log_path
