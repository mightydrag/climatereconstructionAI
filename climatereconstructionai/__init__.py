__version__ = "0.1.0"

# Import train/evaluate lazily. Eagerly importing them here would load the
# submodules when the package is imported, which makes
# `python -m climatereconstructionai.train` emit a RuntimeWarning because the
# module is already in sys.modules prior to being run as __main__.
__all__ = ['train', 'evaluate']


def __getattr__(name):
    if name in ('train', 'evaluate'):
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
