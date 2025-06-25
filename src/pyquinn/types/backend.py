""" Contains helper functions for dynamically loading PyQuinn backends."""

def _load_backend_object(backend: str, import_path: str):
    """Safely loads a backend object"""

    try:
        module_path, attr = import_path.rsplit(".", 1)
        mod = __import__(module_path, fromlist=[attr])
        return getattr(mod, attr)
    except ImportError as e:
        raise ImportError(f"Backend '{backend}' is not available: {e}")
    except AttributeError:
        raise ImportError(
            f"Object '{attr}' not found in module '{module_path}'.")