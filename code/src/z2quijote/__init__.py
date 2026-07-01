"""Independent direct-CDM Quijote active-learning package."""

__all__ = ["Z2Config", "load_config"]


def __getattr__(name: str):
    if name in {"Z2Config", "load_config"}:
        from .config import Z2Config, load_config

        return {"Z2Config": Z2Config, "load_config": load_config}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
