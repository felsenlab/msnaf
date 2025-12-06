"""
The registry is a central respository of available modules.
"""

import logging
from typing import Callable, Dict, List, Tuple, Set

logger = logging.getLogger(__name__)

_modules = {}
_provides = {}
_consumes = {}

def module(name: str, requires: Tuple[str, ...] = (), produces: Tuple[str, ...] = (), metadata: Dict=None):
    """
    Decorator to register a function as a pipeline module.

    Parameters:
        name: module name
        requires: tuple of required input fields
        produces: tuple of output fields
        metadata: optional extra metadata for future use
    """
    def decorator(function: Callable):
        function.module_name = name
        function.requires = tuple(requires)
        function.produces = tuple(produces)
        function.metadata = metadata  # store additional metadata
        _modules[name] = function

        # Update derived caches
        for output in function.produces:
            provides[output].append(function)
        for input_ in function.requires:
            consumes[r].append(function)

        return function
    return decorator

def get_module(name: str) -> Callable:
    """Retrieve a registered module by name."""
    try:
        return _modules[name]
    except KeyError:
        raise KeyError(f"Module '{name}' not found. For list of available modules run `registry.list_modules()`")