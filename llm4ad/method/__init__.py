import os
import importlib
import ast

__all__ = []
_LAZY_ATTRS = {}

# Get the directory of the current package
package_dir = os.path.dirname(__file__)


def _register_lazy_method_classes():
    for subdir_name in os.listdir(package_dir):
        subdir_path = os.path.join(package_dir, subdir_name)
        init_path = os.path.join(subdir_path, '__init__.py')

        if not os.path.isdir(subdir_path) or subdir_name == '__pycache__':
            continue
        if not os.path.exists(init_path):
            continue

        try:
            with open(init_path, encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=init_path)
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue

        for node in tree.body:
            if not isinstance(node, ast.ImportFrom) or node.level != 1 or not node.module:
                continue

            module_name = f'.{subdir_name}.{node.module}'
            for alias in node.names:
                if alias.name == '*':
                    continue
                attr_name = alias.asname or alias.name
                _LAZY_ATTRS.setdefault(attr_name, []).append((module_name, alias.name))
                if attr_name not in __all__:
                    __all__.append(attr_name)


_register_lazy_method_classes()


def __getattr__(name):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    last_error = None
    for module_name, attr_name in reversed(_LAZY_ATTRS[name]):
        try:
            module = importlib.import_module(module_name, package=__name__)
            attr = getattr(module, attr_name)
        except (ImportError, ModuleNotFoundError) as e:
            last_error = e
            continue
        globals()[name] = attr
        return attr

    if last_error:
        raise last_error
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))

def import_all_method_classes_from_subfolders(root_directory: str):
    """
    This function is kept for compatibility but the dynamic import
    is now handled by the package's __init__.py.
    """
    # This function can be left empty or with a pass statement.
    # The dynamic importing is now handled when the package is imported.
    pass
