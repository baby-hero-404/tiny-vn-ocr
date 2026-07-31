"""Registry for benchmark OCR methods."""

from typing import Callable, Dict
import numpy as np

# Method signature: func(image: np.ndarray, doc_type: str) -> dict[str, str]
OCRMethod = Callable[[np.ndarray, str], Dict[str, str]]

_METHODS: Dict[str, OCRMethod] = {}

def register_method(name: str):
    """Decorator to register a new OCR method for benchmarking."""
    def decorator(func: OCRMethod):
        if name in _METHODS:
            raise ValueError(f"Method '{name}' is already registered.")
        _METHODS[name] = func
        return func
    return decorator

def get_method(name: str) -> OCRMethod:
    """Retrieve a registered method by name."""
    if name not in _METHODS:
        raise KeyError(f"Method '{name}' not found. Registered methods: {list(_METHODS.keys())}")
    return _METHODS[name]

def get_all_methods() -> Dict[str, OCRMethod]:
    """Get all registered methods."""
    return _METHODS.copy()
