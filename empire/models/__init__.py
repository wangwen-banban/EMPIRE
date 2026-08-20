"""Model entry points loaded on demand to keep the base import lightweight."""

__all__ = ["EMPIRE_Planner", "load_model"]


def __getattr__(name):
    if name == "EMPIRE_Planner":
        from .vla.empire_planner import EMPIRE_Planner

        return EMPIRE_Planner
    if name == "load_model":
        from .vla_builder import load_model

        return load_model
    raise AttributeError(name)
