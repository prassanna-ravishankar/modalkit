# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Modalkit is a Python framework that sits on top of Modal (modal.com) to provide ML-specific patterns and conveniences for deploying machine learning models. It transforms Modal from infrastructure primitives into a complete ML platform with standardized inference pipelines, configuration-driven deployments, and production-ready features.

**Key Value Proposition**: Modalkit offers standardized ML architecture, configuration-driven deployments, team-friendly workflows, and production features out-of-the-box compared to using raw Modal directly.

## Development Commands

### Environment Setup
```bash
uv sync              # Install virtual environment and dependencies
uv run pre-commit install  # Install pre-commit hooks
```

### Code Quality & Testing
```bash
uv run pre-commit run -a     # Run pre-commit hooks and linting (ruff)
uv run pytest --cov --cov-config=pyproject.toml --cov-report=xml  # Run pytest with coverage
uv run pytest tests/test_specific.py -v  # Run specific test file
uv run pytest tests/test_specific.py::test_function -v  # Run specific test
```

### Building & Documentation
```bash
uvx --from build pyproject-build --installer uv  # Build wheel file
uv run mkdocs serve          # Build and serve documentation locally
uv run mkdocs build -s       # Test documentation build
```

### Package Management
- Uses `uv` as package manager (faster than pip/poetry)
- Dynamic versioning via `setuptools_scm`
- Lock file consistency checking is disabled due to dynamic versioning

## Architecture Overview

### Core Components

**Configuration System** (`modalkit/settings.py`):
- YAML-based configuration with Pydantic models
- Hierarchical settings: `AppSettings` (deployment config) + `ModelSettings` (model-specific config)
- Environment variable overrides supported
- Main config file: `modalkit.yaml` (configurable via `MODALKIT_CONFIG` env var)

**Inference Pipeline** (`modalkit/inference_pipeline.py`):
- `InferencePipeline` abstract class enforces 3-stage pattern: `preprocess()` → `predict()` → `postprocess()`
- All models must inherit from `InferencePipeline`
- Handles volume reloading via `on_volume_reload()` hook
- Type-safe with Pydantic models for input/output

**Modal Integration** (`modalkit/modal_config.py`, `modalkit/model_service.py`):
- `ModalConfig` provides Modal-specific operations and settings translation
- `ModalService` handles Modal deployment patterns and lifecycle
- Automatic cloud bucket mounts, volumes, and secrets management
- Container startup via `@modal.enter()` decorator

**API Layer** (`modalkit/fast_api.py`):
- FastAPI endpoints: `/predict_sync`, `/predict_batch`, `/predict_async`
- Authentication middleware (API key or Modal proxy auth)
- Error handling and request validation
- Queue integration for async processing

**Queue System** (`modalkit/task_queue.py`):
- Protocol-based abstraction supporting multiple backends (SQS, Taskiq)
- Async processing with `send_response_queue()` function

### Configuration Architecture

Configuration flows through this hierarchy:
1. `modalkit.yaml` → Pydantic settings models
2. `Settings` class combines `AppSettings` + `ModelSettings`
3. `ModalConfig` translates settings to Modal-specific parameters
4. `ModalService` uses translated parameters for deployment

### Deployment Flow

1. **Model Definition**: User creates class inheriting from `InferencePipeline`
2. **Configuration**: User defines `modalkit.yaml` with deployment settings
3. **Modal App**: User creates Modal app class inheriting from `ModalService`
4. **Endpoints**: `create_web_endpoints()` generates FastAPI routes
5. **Deployment**: `modal deploy app.py` deploys to Modal infrastructure

## Key Design Patterns

**Protocol-Based Abstractions**: Queue backends use Protocol pattern for multiple implementations

**Settings Translation**: `ModalConfig` translates declarative YAML config to Modal API calls

**Dependency Injection**: FastAPI dependency injection for authentication and validation

**Hook System**: `on_volume_reload()` allows models to react to infrastructure changes

**Type Safety**: Pydantic models throughout for configuration and data validation

## Modal-Specific Concepts

**Cloud Bucket Mounts**: Direct mounting of S3/GCS/R2 buckets via `CloudBucketMount` config

**Volume Management**: Automatic volume mounting and reloading for model artifacts

**Authentication Modes**:
- Custom API key auth with AWS SSM integration
- Modal proxy auth (managed by Modal platform)

**GPU Configuration**: Declarative GPU selection (T4, A10G, A100, H100) via YAML

## Testing Philosophy & Approach

### Test-Driven Development (TDD) - NON-NEGOTIABLE

**"TEST-DRIVEN DEVELOPMENT IS NON-NEGOTIABLE"** - This project strictly follows TDD principles:

1. **Red-Green-Refactor Cycle**: Always write failing tests first, then make them pass, then refactor
2. **No Code Without Tests**: Every new feature, bug fix, or change must have corresponding tests written FIRST
3. **Run Tests Continuously**: Execute `make test` after every change to ensure nothing breaks
4. **Test Coverage**: Maintain high test coverage - all new code must be tested

### Testing Implementation

- **Framework**: Pytest with async support enabled
- **Mocking**: Mock Modal and AWS dependencies using `pytest-mock`
- **Fixtures**: Common setup in `tests/conftest.py` for reusable test components
- **Coverage**: Coverage reporting configured in `pyproject.toml`
- **Organization**: Tests organized by module: `test_<module_name>.py`

### TDD Workflow for Changes

1. **Write failing test** that describes the desired behavior
2. **Run `uv run pytest`** to confirm the test fails (Red)
3. **Write minimal code** to make the test pass (Green)
4. **Run `uv run pytest`** to confirm all tests pass
5. **Refactor** code while keeping tests green
6. **Run `uv run pytest`** again to ensure refactoring didn't break anything

This TDD approach ensures reliability, maintainability, and prevents regressions in this ML deployment framework.

## Documentation Structure

**Code Examples**: Located in `docs/examples/` with 5 comprehensive tutorials:
- Sentiment Analysis (basic)
- LLM Deployment (GPU optimization)
- Computer Vision (multi-task)
- Multi-Modal AI (text/image/audio)
- Real-Time Analytics (stream processing)

**Guides**: In `docs/guide/` covering configuration, deployment, getting started

## Important Implementation Notes

**Error Handling**: Custom exceptions in `modalkit/exceptions.py` with proper error chaining

**Logging**: Structured logging via `python-json-logger` in `modalkit/logger.py`

**Auth Security**: API keys retrieved from AWS SSM in production, never hardcoded

**Batch Processing**: Intelligent batching configuration for GPU efficiency

**Queue Naming**: Functions use `send_response_queue` for backend abstraction

When making changes, ensure compatibility with Modal's API and maintain the configuration-driven approach that's central to Modalkit's value proposition.
