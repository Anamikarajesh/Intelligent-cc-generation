"""Placeholder optical flow helpers."""

from __future__ import annotations


def describe_planned_signals() -> list[str]:
    """Return visual motion signals planned for OpenCV implementation."""

    return [
        "global optical-flow magnitude",
        "localized motion spike",
        "pre/post-event motion delta",
        "camera shake suppression",
    ]
