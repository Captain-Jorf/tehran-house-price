# =========================================================
# Stage 1: builder
# =========================================================
FROM python:3.10.14-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

WORKDIR /build

# Use Iran mirror for Debian apt. Strip bookworm-updates only (Iran mirror has it stale),
# but keep bookworm itself.
RUN set -eux; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's|http://deb.debian.org|https://mirror.iranserver.com|g' /etc/apt/sources.list.d/debian.sources; \
        sed -i 's|http://security.debian.org|https://mirror.iranserver.com|g' /etc/apt/sources.list.d/debian.sources; \
        sed -i 's| bookworm-updates||g' /etc/apt/sources.list.d/debian.sources; \
    fi; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i 's|http://deb.debian.org|https://mirror.iranserver.com|g' /etc/apt/sources.list; \
        sed -i 's|http://security.debian.org|https://mirror.iranserver.com|g' /etc/apt/sources.list; \
        sed -i '/bookworm-updates/d' /etc/apt/sources.list; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Configure pip to use Iran PyPI mirror.
RUN pip config set global.index-url https://mirror-pypi.runflare.com/simple/ \
    && pip config set global.trusted-host mirror-pypi.runflare.com

RUN pip install --user --upgrade pip setuptools wheel

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user -r requirements.txt

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user -e .


# =========================================================
# Stage 2: runtime
# =========================================================
FROM python:3.10.14-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/app/.local/bin:${PATH}" \
    PYTHONPATH="/app/src" \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_LOG_LEVEL=info \
    APP_ENV=prod \
    LOG_LEVEL=INFO

RUN set -eux; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's|http://deb.debian.org|https://mirror.iranserver.com|g' /etc/apt/sources.list.d/debian.sources; \
        sed -i 's|http://security.debian.org|https://mirror.iranserver.com|g' /etc/apt/sources.list.d/debian.sources; \
        sed -i 's| bookworm-updates||g' /etc/apt/sources.list.d/debian.sources; \
    fi; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i 's|http://deb.debian.org|https://mirror.iranserver.com|g' /etc/apt/sources.list; \
        sed -i 's|http://security.debian.org|https://mirror.iranserver.com|g' /etc/apt/sources.list; \
        sed -i '/bookworm-updates/d' /etc/apt/sources.list; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system app \
    && useradd --system --gid app --home-dir /home/app --shell /sbin/nologin app \
    && mkdir -p /home/app /app /app/logs \
    && chown -R app:app /home/app /app

WORKDIR /app

COPY --from=builder --chown=app:app /root/.local /home/app/.local

COPY --chown=app:app src/                       ./src/
COPY --chown=app:app configs/                   ./configs/
COPY --chown=app:app artifacts/models/          ./artifacts/models/
COPY --chown=app:app pyproject.toml README.md   ./

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else sys.exit(1)" \
    || exit 1

CMD ["python", "-m", "tehran_house_price.api"]
