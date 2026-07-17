"""JP Learning Platform package."""

from importlib.metadata import PackageNotFoundError, version

_DISTRIBUTION_NAME = "jp-learning-platform"
_FALLBACK_VERSION = "1.0.0"


def _resolve_version() -> str:
    try:
        return version(_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return _FALLBACK_VERSION


__version__ = _resolve_version()

__all__ = ["__version__"]
