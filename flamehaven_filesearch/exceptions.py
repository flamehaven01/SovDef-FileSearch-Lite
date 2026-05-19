"""
Custom exception classes for FLAMEHAVEN FileSearch

Standardized error handling with proper HTTP status codes and error messages.
"""

from typing import Any, Dict, Optional


class FileSearchException(Exception):
    """Base exception for all FileSearch errors"""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API response"""
        response = {
            "error": self.error_code,
            "message": self.message,
            "status_code": self.status_code,
        }
        if self.details:
            response["details"] = self.details
        return response


# File Upload Errors
class FileUploadError(FileSearchException):
    """Base class for file upload related errors"""

    def __init__(self, message: str, error_code: str = "FILE_UPLOAD_ERROR", **kwargs):
        status_code = kwargs.pop("status_code", 400)
        super().__init__(
            message, status_code=status_code, error_code=error_code, **kwargs
        )


class FileSizeExceededError(FileUploadError):
    """File size exceeds maximum allowed"""

    def __init__(self, file_size: int, max_size: int, filename: Optional[str] = None):
        details = {
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "max_size_mb": max_size,
        }
        if filename:
            details["filename"] = filename

        message = (
            f"File size ({details['file_size_mb']}MB) exceeds maximum allowed "
            f"({max_size}MB)"
        )
        super().__init__(message, error_code="FILE_SIZE_EXCEEDED", details=details)


class InvalidFilenameError(FileUploadError):
    """Invalid or unsafe filename"""

    def __init__(self, filename: str, reason: str = "Invalid filename"):
        details = {"filename": filename, "reason": reason}
        super().__init__(
            f"Invalid filename: {reason}",
            error_code="INVALID_FILENAME",
            details=details,
        )


class UnsupportedFileTypeError(FileUploadError):
    """Unsupported file type"""

    def __init__(self, file_type: str, supported_types: Optional[list] = None):
        details = {"file_type": file_type}
        if supported_types:
            details["supported_types"] = supported_types

        message = f"Unsupported file type: {file_type}"
        super().__init__(message, error_code="UNSUPPORTED_FILE_TYPE", details=details)


class FileProcessingError(FileUploadError):
    """Error processing uploaded file"""

    def __init__(self, message: str, filename: Optional[str] = None):
        if filename and "." in message and " " not in message:
            message, filename = filename, message
        details = {}
        if filename:
            details["filename"] = filename

        super().__init__(message, error_code="FILE_PROCESSING_ERROR", details=details)


# Search Errors
class SearchError(FileSearchException):
    """Base class for search related errors"""

    def __init__(self, message: str, error_code: str = "SEARCH_ERROR", **kwargs):
        status_code = kwargs.pop("status_code", 400)
        super().__init__(
            message, status_code=status_code, error_code=error_code, **kwargs
        )


class EmptySearchQueryError(SearchError):
    """Empty or blank search query"""

    def __init__(self):
        super().__init__(
            "Search query cannot be empty",
            error_code="EMPTY_SEARCH_QUERY",
        )


class InvalidSearchQueryError(SearchError):
    """Invalid search query format or content"""

    def __init__(self, query: str, reason: str = "Invalid query"):
        details = {
            "query": query[:100],
            "reason": reason,
        }  # Limit query length in error
        super().__init__(
            f"Invalid search query: {reason}",
            error_code="INVALID_SEARCH_QUERY",
            details=details,
        )


class SearchTimeoutError(SearchError):
    """Search operation timed out"""

    def __init__(self, timeout_seconds: int):
        details = {"timeout_seconds": timeout_seconds}
        super().__init__(
            f"Search operation timed out after {timeout_seconds} seconds",
            status_code=504,
            error_code="SEARCH_TIMEOUT",
            details=details,
        )


class NoResultsFoundError(FileSearchException):
    """No results found for search query"""

    def __init__(self, query: str):
        details = {"query": query[:100]}
        super().__init__(
            "No results found for the given query",
            status_code=404,
            error_code="NO_RESULTS_FOUND",
            details=details,
        )


# Configuration Errors
class ConfigurationError(FileSearchException):
    """Configuration related errors"""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        details = kwargs.pop("details", {}) or {}
        if field:
            details.setdefault("field", field)
        status_code = kwargs.pop("status_code", 500)
        error_code = kwargs.pop("error_code", "CONFIGURATION_ERROR")
        super().__init__(
            message,
            status_code=status_code,
            error_code=error_code,
            details=details,
            **kwargs,
        )


class MissingAPIKeyError(ConfigurationError):
    """API key is missing"""

    def __init__(self):
        super().__init__(
            (
                "API key is required for remote mode. "
                "Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            ),
            error_code="MISSING_API_KEY",
        )


class InvalidAPIKeyError(ConfigurationError):
    """API key is invalid"""

    def __init__(self, reason: Optional[str] = None):
        details = {}
        if reason:
            details["reason"] = reason

        message = "Invalid API key"
        if reason:
            message += f": {reason}"

        super().__init__(
            message,
            status_code=401,
            error_code="INVALID_API_KEY",
            details=details,
        )


# Rate Limiting Errors
class RateLimitExceededError(FileSearchException):
    """Rate limit exceeded"""

    def __init__(
        self,
        limit: Any,
        window: Optional[int] = None,
        retry_after: Optional[int] = None,
    ):
        if isinstance(limit, str) and window is None:
            details = {}
            if retry_after:
                details["retry_after_seconds"] = retry_after
            super().__init__(
                limit,
                status_code=429,
                error_code="RATE_LIMIT_EXCEEDED",
                details=details,
            )
            return

        if window is None:
            window = 60
        details = {
            "limit": limit,
            "window_seconds": window,
        }
        if retry_after:
            details["retry_after_seconds"] = retry_after

        message = f"Rate limit exceeded: {limit} requests per {window} seconds"
        super().__init__(
            message,
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
            details=details,
        )


# Validation Errors
class ValidationError(FileSearchException):
    """Input validation error"""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if field:
            details["field"] = field

        super().__init__(
            message,
            status_code=422,
            error_code="VALIDATION_ERROR",
            details=details,
            **{k: v for k, v in kwargs.items() if k != "details"},
        )


# Service Errors
class ServiceUnavailableError(FileSearchException):
    """Service temporarily unavailable"""

    def __init__(self, service: str, reason: Optional[str] = None):
        details = {"service": service}
        if reason:
            details["reason"] = reason

        message = f"Service '{service}' is temporarily unavailable"
        if reason:
            message += f": {reason}"

        super().__init__(
            message,
            status_code=503,
            error_code="SERVICE_UNAVAILABLE",
            details=details,
        )


class ExternalAPIError(FileSearchException):
    """External API call failed"""

    def __init__(self, api_name: str, reason: str, status_code: Optional[int] = None):
        details = {"api": api_name, "reason": reason}
        if status_code:
            details["api_status_code"] = status_code

        super().__init__(
            f"External API '{api_name}' error: {reason}",
            status_code=502,
            error_code="EXTERNAL_API_ERROR",
            details=details,
        )


# Resource Errors
class ResourceNotFoundError(FileSearchException):
    """Requested resource not found"""

    def __init__(self, resource_type: str, resource_id: str):
        details = {"resource_type": resource_type, "resource_id": resource_id}
        super().__init__(
            f"{resource_type} not found: {resource_id}",
            status_code=404,
            error_code="RESOURCE_NOT_FOUND",
            details=details,
        )


class ResourceConflictError(FileSearchException):
    """Resource already exists or conflict"""

    def __init__(
        self, resource_type: str, resource_id: str, reason: Optional[str] = None
    ):
        details = {"resource_type": resource_type, "resource_id": resource_id}
        if reason:
            details["reason"] = reason

        message = f"{resource_type} conflict: {resource_id}"
        if reason:
            message += f" - {reason}"

        super().__init__(
            message,
            status_code=409,
            error_code="RESOURCE_CONFLICT",
            details=details,
        )


# Internal Errors
class InternalServerError(FileSearchException):
    """Internal server error"""

    def __init__(self, message: str = "An internal error occurred", **kwargs):
        super().__init__(
            message,
            status_code=500,
            error_code="INTERNAL_SERVER_ERROR",
            **kwargs,
        )


# Helper function to convert exceptions to HTTP responses
def exception_to_response(exc: Exception) -> Dict[str, Any]:
    """
    Convert any exception to a standardized HTTP response dictionary

    Args:
        exc: Exception instance

    Returns:
        Dictionary with error details suitable for HTTP response
    """
    if isinstance(exc, FileSearchException):
        return exc.to_dict()

    # Handle standard Python exceptions
    if isinstance(exc, ValueError):
        return {
            "error": "VALIDATION_ERROR",
            "message": str(exc),
            "status_code": 422,
        }
    elif isinstance(exc, FileNotFoundError):
        return {
            "error": "FILE_NOT_FOUND",
            "message": str(exc),
            "status_code": 404,
        }
    elif isinstance(exc, PermissionError):
        return {
            "error": "PERMISSION_DENIED",
            "message": "Permission denied",
            "status_code": 403,
        }
    elif isinstance(exc, TimeoutError):
        return {
            "error": "TIMEOUT",
            "message": "Operation timed out",
            "status_code": 504,
        }
    else:
        # Unknown exception - return generic internal error
        return {
            "error": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "status_code": 500,
        }
