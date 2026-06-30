import os
import importlib
import ast

__all__ = []
_LAZY_ATTRS = {}

# Get the directory of the current package
package_dir = os.path.dirname(__file__)
root_package = __name__


def _register_lazy_evaluation_classes():
    # Recursively find evaluation.py files without importing task dependencies.
    for dirpath, _, filenames in os.walk(package_dir):
        if 'evaluation.py' not in filenames:
            continue

        rel_path = os.path.relpath(dirpath, package_dir)
        if rel_path == '.':
            submodule_suffix = 'evaluation'
        else:
            submodule_suffix = rel_path.replace(os.path.sep, '.') + '.evaluation'

        evaluation_path = os.path.join(dirpath, 'evaluation.py')
        try:
            with open(evaluation_path, encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=evaluation_path)
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue

        module_name = f'.{submodule_suffix}'
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            _LAZY_ATTRS[node.name] = (module_name, node.name)
            if node.name not in __all__:
                __all__.append(node.name)


_register_lazy_evaluation_classes()


def __getattr__(name):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_ATTRS[name]
    module = importlib.import_module(module_name, package=root_package)
    attr = getattr(module, attr_name)
    globals()[name] = attr
    return attr


def __dir__():
    return sorted(set(globals()) | set(__all__))

def import_all_evaluation_classes(root_directory):
    """
    This function is kept for compatibility but the dynamic import
    is now handled by the package's __init__.py.
    """
    pass
