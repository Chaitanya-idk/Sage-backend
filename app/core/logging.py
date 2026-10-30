import logging
import sys
import re
from typing import Optional


class SensitiveDataFilter(logging.Filter):
    """Filter that masks sensitive tokens (HF_TOKEN, API keys) from log messages."""

    PATTERNS = [
        (re.compile(r"(hf_[a-zA-Z0-9]{30,})"), "[REDACTED_HF_TOKEN]"),
        (re.compile(r"(Bearer\s+)([a-zA-Z0-9_\-\.]{8,})", re.IGNORECASE), r"\1[REDACTED_KEY]"),
        (re.compile(r"(api[_-]?key[\"'\s:=]+)([a-zA-Z0-9_\-]{8,})", re.IGNORECASE), r"\1[REDACTED_KEY]"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            for pattern, repl in self.PATTERNS:
                record.msg = pattern.sub(repl, record.msg)
        return True


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure structured logging for the SAGE backend application."""
    root_logger = logging.getLogger()
    
    # Avoid duplicate handlers if re-initialized
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    handler.addFilter(SensitiveDataFilter())

    root_logger.addHandler(handler)

    # Suppress verbose third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    return root_logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a named logger with sensitive data protection."""
    logger = logging.getLogger(name or "sage")
    return logger

