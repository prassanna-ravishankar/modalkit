from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, Body, Depends, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from modalkit.iomodel import AsyncInputModel, AsyncOutputModel, SyncInputModel


def create_app(
    input_model: type[BaseModel],
    output_model: type[BaseModel],
    dependencies: list[Callable[..., Any] | None],
    router_dependency: Callable[..., Any] | None,
    sync_fn: Callable[[str, BaseModel], Awaitable[BaseModel]],
    async_fn: Callable[[str, BaseModel], Awaitable[AsyncOutputModel]],
    default_model_name: str | None = None,
) -> FastAPI:
    """
    Creates and configures a FastAPI application with synchronous and asynchronous predict endpoints.
    Routes rely on Modal proxy auth by default, with optional router dependencies.

    Args:
        input_model: Pydantic model defining the input schema for predict requests.
        output_model: Pydantic model defining the output schema for predict responses.
        dependencies: List of global dependencies for the FastAPI application.
        router_dependency: Optional dependency for router-level functionality.
        sync_fn: Synchronous predict function.
        async_fn: Asynchronous predict function, must return job_id.
        default_model_name: Default model name for requests that don't specify one.
            When set, model_name becomes an optional query parameter.

    Returns:
        FastAPI: Configured FastAPI application with predict routes.

    Routes:
        - `/health` (GET): Health check endpoint (unauthenticated).
        - `/predict_sync` (POST): Synchronous predict endpoint.
        - `/predict_async` (POST): Asynchronous predict endpoint.
    """
    fastapi_deps = [Depends(dep) for dep in dependencies if dep]
    app = FastAPI(dependencies=fastapi_deps)

    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Any, exc: ValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    # Create router with optional dependency, otherwise use only Modal proxy auth
    router_dependencies = [Depends(router_dependency)] if router_dependency is not None else []
    authenticated_router = APIRouter(dependencies=router_dependencies)

    # Define Body annotation once to avoid linting issues
    _body_annotation = Body(...)

    @authenticated_router.post("/predict_sync", response_model=output_model)
    async def predict_sync(request: Any = _body_annotation, model_name: str = default_model_name or "") -> Any:
        parsed_request: BaseModel = input_model.model_validate(request)
        wrapped_input = SyncInputModel(message=parsed_request)
        return await sync_fn(model_name, wrapped_input)

    @authenticated_router.post("/predict_async", response_model=AsyncOutputModel)
    async def predict_async(request: Any = _body_annotation, model_name: str = default_model_name or "") -> Any:
        parsed_request: AsyncInputModel = AsyncInputModel.model_validate(request)
        return await async_fn(model_name, parsed_request)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(authenticated_router)

    return app
