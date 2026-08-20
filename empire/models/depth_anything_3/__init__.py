"""Depth Anything 3 runtime subset bundled for EMPIRE depth conditioning."""

__all__ = ["DepthAnything3"]


def __getattr__(name):
    if name == "DepthAnything3":
        from .api import DepthAnything3

        return DepthAnything3
    raise AttributeError(name)
