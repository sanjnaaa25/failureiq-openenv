"""Thin wrapper for OpenEnv validator entrypoint."""

from app import app
import uvicorn

__all__ = ["app", "main"]


def main() -> None:
    """Entry point required by validator for multi-mode deployment."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
