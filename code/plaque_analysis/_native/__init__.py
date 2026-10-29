"""Native measurement implementations used by the public interfaces."""
import importlib as _importlib


_CORE_MODULES = frozenset(("authorization", "human_measurements", "fluorescence_geometry",
                           "texture", "lamp1", "registration", "neighborhoods", "observed_windows",
                           "abeta", "colocalization"))


class _UnavailableCore:
    def __getattr__(self, name):
        raise ModuleNotFoundError(
            "This computation requires the private plaque-analysis-core package. "
            "Request the matching core package and signed license from the project owner.",
            name="plaque_analysis._native")


def __getattr__(name):
    if name not in _CORE_MODULES:
        raise AttributeError(name)
    try:
        return _importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        if exc.name != f"{__name__}.{name}":
            raise
        return _UnavailableCore()
