from pathlib import Path
from loguru import logger

def setup_logger(verbose: bool = True):
    """
    Configure Loguru. If verbose=False, only WARNING/ERROR/SUCCESS in terminal.
    File logs are always DEBUG-level with rotation.
    """
    # base paths
    project_root = Path(__file__).resolve().parents[1]
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    # reset handlers
    logger.remove()

    # terminal sink
    level = "INFO" if verbose else "WARNING"
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        colorize=True,        # helps on Windows Terminal/PowerShell
    )

    # file sink
    logger.add(
        log_dir / "pft.log",
        rotation="1 week",
        retention="1 month",
        compression="zip",
        level="DEBUG",
        encoding="utf-8",
    )

    return logger

__all__ = ["setup_logger", "logger"]
