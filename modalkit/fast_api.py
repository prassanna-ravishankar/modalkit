from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from fastapi import APIRouter, Body, Depends, FastAPI
from pydantic import BaseModel

from modalkit.iomodel import AsyncInputModel, AsyncOutputModel, SyncInputModel

T_input = TypeVar("T_input", bound=BaseModel)
T_output = TypeVar("T_output", bound=BaseModel)


def create_app(
    input_model: type[BaseModel],
    output_model: type[BaseModel],
    dependencies: list[Callable[..., Any] | None],
    router_dependency: Callable[..., Any] | None,
    sync_fn: Callable[[str, BaseModel], Awaitable[BaseModel]],
    async_fn: Callable[[str, BaseModel], Awaitable[AsyncOutputModel]],
) -> FastAPI:
    """
    Creates and configures a FastAPI application with synchronous and asynchronous predict endpoints.
    Routes rely on Modal proxy auth by default, with optional router dependencies.

    Args:
        input_model (BaseModel): Pydantic model defining the input schema for predict requests.
        output_model (BaseModel): Pydantic model defining the output schema for predict responses.
        dependencies (list[Optional[Callable[..., Any]]]): List of global dependencies for the FastAPI application.
        router_dependency (Optional[Callable[..., Any]]): Optional dependency for router-level functionality.
                                                          If None, routes use only Modal proxy auth.
        sync_fn (Callable[[str, BaseModel], Awaitable[BaseModel]]): Synchronous predict function.
        async_fn (Callable[[str, BaseModel], Awaitable[AsyncOutputModel]]): Asynchronous
            predict function, must return job_id

    Returns:
        FastAPI: Configured FastAPI application with predict routes.

    Routes:
        - `/predict_sync` (POST): Synchronous predict endpoint.
            Processes requests individually.
        - `/predict_async` (POST): Asynchronous predict endpoint.
            Processes requests using batching based on batch_config settings.
    """
    fastapi_deps = [Depends(dep) for dep in dependencies if dep]
    app = FastAPI(dependencies=fastapi_deps)

    # Create router with optional dependency, otherwise use only Modal proxy auth
    router_dependencies = [Depends(router_dependency)] if router_dependency is not None else []
    authenticated_router = APIRouter(dependencies=router_dependencies)

    # Define Body annotation once to avoid linting issues
    _body_annotation = Body(...)

    @authenticated_router.post("/predict_sync", response_model=output_model)
    async def predict_sync(model_name: str, request: Any = _body_annotation) -> Any:
        # Parse the request using the provided input model
        parsed_request: BaseModel = input_model.model_validate(request)
        wrapped_input = SyncInputModel(message=parsed_request)
        return await sync_fn(model_name, wrapped_input)

    @authenticated_router.post("/predict_async", response_model=AsyncOutputModel)
    async def predict_async(model_name: str, request: Any = _body_annotation) -> Any:
        # Parse as AsyncInputModel
        parsed_request: AsyncInputModel = AsyncInputModel.model_validate(request)
        return await async_fn(model_name, parsed_request)

    app.include_router(authenticated_router)

    return app
