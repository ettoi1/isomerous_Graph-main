"""Lightweight logging helpers.

用于记录synthetic冒烟测试和真实训练时的关键信息。"""
from __future__ import annotations

import datetime
from typing import Any


def log(message: str, *args: Any) -> None:
    """Print timestamped log message."""

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if args:
        message = message.format(*args)
    print(f"[{timestamp}] {message}")


if __name__ == "__main__":
    log("Logging utility ready. Value={}.", 123)
