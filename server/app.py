from __future__ import annotations

from api.main import app, main as api_main

__all__ = ["app", "main"]


def main() -> None:
    api_main()


if __name__ == "__main__":
    main()
