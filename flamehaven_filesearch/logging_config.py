"""
Structured JSON logging configuration for FLAMEHAVEN FileSearch

Production-ready logging with JSON format for log aggregation systems.
"""

import logging
import sys
from datetime import datetime, timezone

from . import __version__

try:
    from pythonjsonlogger import jsonlogger

    _JSONLOGGER_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback when dependency missing
    _JSONLOGGER_AVAILABLE = False
    jsonlogger = None


if _JSONLOGGER_AVAILABLE:

    class CustomJsonFormatter(jsonlogger.JsonFormatter):
        """
        Custom JSON formatter with additional fields

        Adds request_id, service_name, and environment to all log records.
        """

        def add_fields(self, log_record, record, message_dict):
            super(CustomJsonFormatter, self).add_fields(
                log_record, record, message_dict
            )

            # Add service identification
            log_record["service"] = "flamehaven-filesearch"
            log_record["version"] = __version__

            # Add request ID if available
            if hasattr(record, "request_id"):
                log_record["request_id"] = record.request_id

            # Add environment (from env var or default to 'development')
            import os

            log_record["environment"] = os.getenv("ENVIRONMENT", "development")

            # Ensure timestamp is present
            if "timestamp" not in log_record:
                log_record["timestamp"] = (
                    datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                )

            # Add level name
            log_record["level"] = record.levelname

else:

    class CustomJsonFormatter(logging.Formatter):
        """Fallback JSON formatter when python-json-logger is unavailable."""

        def format(self, record):
            import json
            import os

            payload = {
                "timestamp": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "service": "flamehaven-filesearch",
                "version": __version__,
                "environment": os.getenv("ENVIRONMENT", "development"),
            }
            if hasattr(record, "request_id"):
                payload["request_id"] = record.request_id
            if record.exc_info:
                payload["exc_info"] = self.formatException(record.exc_info)
            return json.dumps(payload)


def setup_json_logging(log_level=logging.INFO, **kwargs):
    """
    Setup structured JSON logging for the application

    Args:
        log_level: Logging level (default: INFO). Deprecated in favor of keyword
            "level".
    """
    # Support logging.basicConfig-style signatures (level=...)
    effective_level = kwargs.get("level", log_level)

    # Create JSON formatter (or fallback)
    if _JSONLOGGER_AVAILABLE:
        formatter = CustomJsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s %(request_id)s "
            "%(service)s %(version)s %(environment)s",
            rename_fields={
                "levelname": "level",
                "name": "logger",
                "asctime": "timestamp",
            },
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        formatter = CustomJsonFormatter()

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(effective_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add handler to stdout
    json_handler = logging.StreamHandler(sys.stdout)
    json_handler.setFormatter(formatter)
    root_logger.addHandler(json_handler)

    # Set level for specific loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)

    return root_logger


def setup_logging(log_level=logging.INFO, **kwargs):
    """Backward-compatible logging entry point used by older tests/callers."""
    environment = kwargs.pop("environment", None) or kwargs.pop("mode", None)
    if environment == "development":
        return setup_development_logging(log_level=log_level, **kwargs)
    return setup_json_logging(log_level=log_level, **kwargs)


def setup_development_logging(log_level=logging.INFO, **kwargs):
    """
    Setup human-readable logging for development

    Args:
        log_level: Logging level (default: INFO). Deprecated in favor of keyword
            "level".
    """
    effective_level = kwargs.get("level", log_level)
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(effective_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return root_logger


def get_logger_with_request_id(name: str, request_id: str = None):
    """
    Get logger with request ID context

    Args:
        name: Logger name
        request_id: Request ID to include in logs

    Returns:
        Logger adapter with request ID
    """
    logger = logging.getLogger(name)

    if request_id:
        has_filter = any(
            isinstance(existing, RequestIdFilter)
            and getattr(existing, "request_id", None) == request_id
            for existing in logger.filters
        )
        if not has_filter:
            logger.addFilter(RequestIdFilter(request_id))
        return logging.LoggerAdapter(logger, {"request_id": request_id})

    return logger


class RequestIdFilter(logging.Filter):
    """
    Logging filter to add request ID to log records
    """

    def __init__(self, request_id="N/A"):
        super().__init__()
        self.request_id = request_id

    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = self.request_id
        return True


# Context manager for request-scoped logging
class RequestLoggingContext:
    """
    Context manager for request-scoped logging with request ID

    Usage:
        with RequestLoggingContext(request_id):
            logger.info("Processing request")  # Includes request_id
    """

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.filter = RequestIdFilter(request_id)
        self.logger = logging.getLogger()

    def __enter__(self):
        self.logger.addFilter(self.filter)
        return self

    def __exit__(self, exc_type, exc_val, _exc_tb):
        self.logger.removeFilter(self.filter)


# Example log entries for different scenarios
EXAMPLE_LOGS = {
    "info": {
        "timestamp": "2025-11-13T12:00:00Z",
        "level": "INFO",
        "logger": "flamehaven_filesearch.api",
        "message": "File uploaded successfully",
        "request_id": "a1b2c3d4-5678-90ab-cdef",
        "service": "flamehaven-filesearch",
        "version": __version__,
        "environment": "production",
        "filename": "document.pdf",
        "size_mb": 2.5,
    },
    "error": {
        "timestamp": "2025-11-13T12:00:01Z",
        "level": "ERROR",
        "logger": "flamehaven_filesearch.api",
        "message": "File upload failed",
        "request_id": "a1b2c3d4-5678-90ab-cdef",
        "service": "flamehaven-filesearch",
        "version": __version__,
        "environment": "production",
        "error": "FileSizeExceededError",
        "error_message": "File size exceeds maximum",
        "filename": "large_file.pdf",
        "size_mb": 100.0,
    },
    "warning": {
        "timestamp": "2025-11-13T12:00:02Z",
        "level": "WARNING",
        "logger": "flamehaven_filesearch.api",
        "message": "Rate limit approaching",
        "request_id": "a1b2c3d4-5678-90ab-cdef",
        "service": "flamehaven-filesearch",
        "version": __version__,
        "environment": "production",
        "endpoint": "/api/upload/single",
        "requests_count": 8,
        "limit": 10,
    },
}
