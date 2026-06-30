import os
import importlib
import ast

__all__ = []
_LAZY_ATTRS = {}

# Get the directory of the current package
package_dir = os.path.dirname(__file__)


def _register_lazy_llm_classes():
    for filename in os.listdir(package_dir):
        if not filename.endswith('.py') or filename == '__init__.py':
            continue

        module_name = filename[:-3]
        module_path = f'.{module_name}'
        file_path = os.path.join(package_dir, filename)

        try:
            with open(file_path, encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=file_path)
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue

        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            _LAZY_ATTRS[node.name] = (module_path, node.name)
            if node.name not in __all__:
                __all__.append(node.name)


_register_lazy_llm_classes()


def __getattr__(name):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_ATTRS[name]
    module = importlib.import_module(module_name, package=__name__)
    attr = getattr(module, attr_name)
    globals()[name] = attr
    return attr


def __dir__():
    return sorted(set(globals()) | set(__all__))

def import_all_llm_classes_from_subfolders(root_directory):
    """
    This function is kept for compatibility but the dynamic import
    is now handled by the package's __init__.py.
    """
    # This function can be left empty or with a pass statement.
    # The dynamic importing is now handled when the package is imported.
    pass
