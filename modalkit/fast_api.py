from collections.abc import Awaitable
from typing import Any, Callable, Optional

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel

from modalkit.iomodel import AsyncInputModel, AsyncOutputModel, SyncInputModel


def create_app(
    input_model: BaseModel,
    output_model: BaseModel,
    dependencies: list[Optional[Callable[..., Any]]],
    auth_middleware: Optional[Callable[..., Any]],
    sync_fn: Callable[[str, BaseModel, dict], Awaitable[BaseModel]],
    async_fn: Callable[[str, BaseModel, dict], Awaitable[AsyncOutputModel]],
):
    """
    Creates and configures a FastAPI application with synchronous and asynchronous predict endpoints.
    Routes are authenticated when auth_middleware is provided, otherwise rely on Modal proxy auth.

    Args:
        input_model (BaseModel): Pydantic model defining the input schema for predict requests.
        output_model (BaseModel): Pydantic model defining the output schema for predict responses.
        dependencies (list[Optional[Callable[..., Any]]]): List of global dependencies for the FastAPI application.
        auth_middleware (Optional[Callable[..., Any]]): Middleware dependency to enforce authentication for specific routes.
                                                        If None, routes are unprotected (rely on Modal proxy auth).
        sync_fn (Callable[[str, BaseModel, dict], BaseModel]): Synchronous predict function.
        async_fn (Callable[[str, BaseModel, dict], AsyncOutputModel]): Asynchronous predict function, must return job_id

    Returns:
        FastAPI: Configured FastAPI application with predict routes.

    Routes:
        - `/predict_sync` (POST): Synchronous predict endpoint.
            Processes requests individually.
        - `/predict_async` (POST): Asynchronous predict endpoint.
            Processes requests using batching based on batch_config settings.
    """
    app = FastAPI(dependencies=dependencies)

    # Create router with auth middleware if provided, otherwise no authentication at FastAPI level
    router_dependencies = [auth_middleware] if auth_middleware is not None else []
    authenticated_router = APIRouter(dependencies=router_dependencies)

    @authenticated_router.post("/predict_sync")
    async def predict_sync(model_name: str, request: input_model) -> output_model:
        wrapped_input = SyncInputModel(message=request)
        return await sync_fn(model_name, wrapped_input)

    @authenticated_router.post("/predict_async")
    async def predict_async(model_name: str, request: AsyncInputModel[input_model]) -> AsyncOutputModel:
        return await async_fn(model_name, request)

    app.include_router(authenticated_router)

    return app
