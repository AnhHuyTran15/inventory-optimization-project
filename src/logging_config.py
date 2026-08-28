"""
logging_config.py
==================
Centralized logging setup for the whole pipeline (notebooks, src modules,
and the Streamlit app all import from here, so every run is traceable
in one place instead of scattered print() statements).

Usage in any notebook or script:

    from src.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Starting Layer A DP solve for store=%s item=%s", store, item)

Design choices (documented so a reviewer / future-you knows why):
  - One log file per calendar day (logs/pipeline_YYYY-MM-DD.log), capped at
    5MB with 3 backups (pipeline_YYYY-MM-DD.log.1, .2, .3) so a long-lived
    deployed process can't grow the file without bound.
  - Also logs to stdout (console) at the same time, so notebooks show
    output live AND persist it to disk.
  - A separate JSON-lines run log (logs/runs.jsonl) captures structured
    events (layer name, store, item, key metrics like s/S/violations) for
    later analysis - e.g. "how many of the 500 pairs converged in under
    500 iterations" becomes a one-line pandas read_json call instead of
    grepping text logs.
  - File logging is best-effort: get_logger() and log_run_event() are called
    at import time by every src module (including the Streamlit app's import
    chain), so on a deploy target with a read-only or ephemeral filesystem a
    write failure here must not crash the app - it falls back to
    console-only logging instead.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

try:
    LOG_DIR.mkdir(exist_ok=True)
    _LOG_DIR_WRITABLE = True
except OSError:
    _LOG_DIR_WRITABLE = False

_CONFIGURED = False


def get_logger(name: str = "inventory_pipeline") -> logging.Logger:
    """Return a logger configured to write to console + a size-capped daily file."""
    global _CONFIGURED

    logger = logging.getLogger(name)

    if not _CONFIGURED:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        root = logging.getLogger()
        root.setLevel(logging.INFO)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

        if _LOG_DIR_WRITABLE:
            log_file = LOG_DIR / f"pipeline_{datetime.now(timezone.utc):%Y-%m-%d}.log"
            try:
                file_handler = RotatingFileHandler(
                    log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
                )
                file_handler.setFormatter(formatter)
                root.addHandler(file_handler)
            except OSError:
                logger.warning("Could not open %s for writing; logging to console only.", log_file)

        _CONFIGURED = True

    return logger


def log_run_event(event_type: str, **fields) -> None:
    """
    Append a structured event to logs/runs.jsonl.

    Example:
        log_run_event(
            "layer_a_solved",
            store=6, item=5, s=16, S=484, iterations=1028, converged=True,
        )

    Kept deliberately separate from the text logger above: text logs are
    for humans reading during a run, this JSONL file is for later
    programmatic analysis (e.g. pd.read_json('logs/runs.jsonl', lines=True)).
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        **fields,
    }
    if not _LOG_DIR_WRITABLE:
        return
    runs_file = LOG_DIR / "runs.jsonl"
    try:
        with open(runs_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        get_logger(__name__).warning("Could not append to %s.", runs_file)
