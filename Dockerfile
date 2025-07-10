FROM python:3.11-slim-bookworm

RUN apt-get update && \
    apt-get install -yq --no-install-recommends git openssh-client && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.10 /uv /uvx /bin/
ENV UV_PROJECT_ENVIRONMENT=/usr/local
ENV UV_LINK_MODE=copy

# Change the working directory to the `app` directory
WORKDIR /app
# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

# Copy the project into the image
ADD . /app

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen
