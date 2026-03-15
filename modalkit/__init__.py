"""Modalkit - Python framework for deploying ML models on Modal."""

__version__ = "0.1.0"

from modalkit.exceptions import AuthConfigError, BackendError, DependencyError, TypeValidationError
from modalkit.inference_pipeline import InferencePipeline
from modalkit.iomodel import AsyncInputModel, AsyncOutputModel, InferenceOutputModel, SyncInputModel
from modalkit.modal_config import ModalConfig
from modalkit.modal_service import ModalService, create_web_endpoints
from modalkit.settings import Settings
from modalkit.task_queue import QueueBackend

__all__ = [
    "AsyncInputModel",
    "AsyncOutputModel",
    "AuthConfigError",
    "BackendError",
    "DependencyError",
    "InferenceOutputModel",
    "InferencePipeline",
    "ModalConfig",
    "ModalService",
    "QueueBackend",
    "Settings",
    "SyncInputModel",
    "TypeValidationError",
    "create_web_endpoints",
]
