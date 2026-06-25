FROM python:3.12-slim

# Install uv for fast, reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY cobaiter ./cobaiter
RUN uv sync --no-dev --frozen || uv sync --no-dev

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "cobaiter"]
