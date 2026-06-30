"""CLI entry point for running the FastAPI server with uvicorn.

This module wraps uvicorn so users can start the API with a single
command and consistent defaults across local dev and deployment.
"""

from __future__ import annotations

import argparse
import os

import uvicorn

from tehran_house_price.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_LOG_LEVEL = "info"
APP_IMPORT_PATH = "tehran_house_price.api.app:app"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments with environment variable fallbacks."""
    parser = argparse.ArgumentParser(
        prog="tehran_house_price.api",
        description="Run the Tehran House Price prediction API server.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("API_HOST", DEFAULT_HOST),
        help="Network interface to bind to (default: %(default)s).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("API_PORT", str(DEFAULT_PORT))),
        help="Port to listen on (default: %(default)s).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=os.getenv("API_RELOAD", "").lower() in {"1", "true", "yes"},
        help="Enable auto-reload for local development.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("API_WORKERS", "1")),
        help="Number of worker processes (default: %(default)s, ignored with --reload).",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("API_LOG_LEVEL", DEFAULT_LOG_LEVEL),
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Uvicorn log level (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Start the uvicorn server with parsed configuration."""
    args = _parse_args(argv)

    logger.info(
        "starting uvicorn | host=%s | port=%s | reload=%s | workers=%s | log_level=%s",
        args.host,
        args.port,
        args.reload,
        args.workers,
        args.log_level,
    )

    uvicorn.run(
        APP_IMPORT_PATH,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
