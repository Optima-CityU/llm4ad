__version__ = '1.0.0'

import importlib

__all__ = ['base', 'method', 'task', 'tools', 'profiler', 'llm']

_LAZY_MODULES = {
    'base': '.base',
    'method': '.method',
    'task': '.task',
    'tools': '.tools',
    'profiler': '.tools.profiler',
    'llm': '.tools.llm',
}


def __getattr__(name):
    if name not in _LAZY_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(_LAZY_MODULES[name], package=__name__)
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(globals()) | set(__all__))
