"""Load fitted atlases across the package's module-layout transition."""

from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
import sys
from threading import RLock
from types import ModuleType


_LOAD_LOCK = RLock()


@contextmanager
def _legacy_model_modules():
    """Resolve only the former atlas and transformation pickle module names."""
    modules = {
        "atlas": import_module("plaque_analysis.models.atlas"),
        "preprocessing": import_module("plaque_analysis.models.preprocessing"),
    }
    package = ModuleType("figure5_core")
    package.__path__ = []
    aliases = {"figure5_core": package}
    for name, module in modules.items():
        setattr(package, name, module)
        aliases[f"figure5_core.{name}"] = module
    missing = object()
    with _LOAD_LOCK:
        previous = {name: sys.modules.get(name, missing) for name in aliases}
        sys.modules.update(aliases)
        try:
            yield
        finally:
            for name, module in previous.items():
                if module is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module


def load_atlas(model_path):
    """Read a fitted atlas using stable package paths and legacy name aliases."""
    import joblib

    with _legacy_model_modules():
        return joblib.load(Path(model_path))
