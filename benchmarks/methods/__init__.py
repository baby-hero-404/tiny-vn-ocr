"""Initialize benchmark methods.

Importing this module will automatically import and register all methods inside.
"""

import importlib
import pkgutil
from pathlib import Path

# Automatically import all modules in this directory so their @register_method decorators run
for _, module_name, _ in pkgutil.iter_modules([str(Path(__file__).parent)]):
    importlib.import_module(f"benchmarks.methods.{module_name}")
