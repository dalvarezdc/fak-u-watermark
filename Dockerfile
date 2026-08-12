# uv-based image — https://docs.astral.sh/uv/guides/integration/docker/
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# System deps for building wheels when needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Project metadata + lock first for better layer caching
COPY pyproject.toml uv.lock README.md ./
COPY packages ./packages
COPY api ./api
COPY ui ./ui
COPY cli ./cli

# Install into project .venv (frozen when lock is present)
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

RUN uv sync --frozen --no-dev

EXPOSE 8000 7860
ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
