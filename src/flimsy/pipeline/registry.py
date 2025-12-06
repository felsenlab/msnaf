"""
The registry is a central respository of available modules.
"""

import logging

logger = logging.getLogger(__name__)

_modules = {}
_provides = {}
_consumes = {}

def module(name: str, requires: Tuple[str, ...] = (), produces: Tuple[str, ...] = (), metadata=None):
    """
    Decorator to register a function as a pipeline module.

    Parameters:
        name: module name
        requires: tuple of required input fields
        produces: tuple of output fields
        **metadata: optional extra metadata for future use
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