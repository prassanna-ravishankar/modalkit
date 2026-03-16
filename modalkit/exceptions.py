"""Custom exceptions for modalkit."""


class DependencyError(ImportError):
    """Raised when required dependencies are missing."""


class TypeValidationError(TypeError):
    """Raised when type validation fails."""


class BackendError(ValueError):
    """Raised when backend configuration is invalid."""
