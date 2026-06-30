import importlib

__all__ = ['llm', 'profiler']

_LAZY_MODULES = {
    'llm': '.llm',
    'profiler': '.profiler',
}


def __getattr__(name):
    if name not in _LAZY_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(_LAZY_MODULES[name], package=__name__)
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(globals()) | set(__all__))
