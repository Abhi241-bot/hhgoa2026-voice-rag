"""
Structured JSON logging setup using structlog.
All pipeline components import `get_logger` from here.
"""

import logging
import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog with JSON output for structured, machine-parseable logs."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper(), logging.INFO),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a named structured logger bound to a component name."""
    configure_logging()
    return structlog.get_logger(name)
