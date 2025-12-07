"""
The registry is a central respository of available modules.
"""
import sys
import pkgutil
import logging
import traceback
import importlib
from pathlib import Path
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Set

import flimsy.modules

logger = logging.getLogger(__name__)

_modules = {}
_provides = defaultdict(list)
_consumes = defaultdict(list)

def _ensure_list(value, field_name, module_name):
    if isinstance(value, list):
        return value
    if value is None:
        raise TypeError(f'{field_name} must be a list, but got NoneType instead')
    if value == []:
        logger.warning(f'{field_name} got an empty list -- this is unlikely to be correct, modules should typically consume some input to produce some output')
    # Reject strings explicitly
    if isinstance(value, str):
        raise TypeError(
            f"In module '{module_name}': the '{field_name}' field must be a list, "
            f"but a string was provided. Did you mean ['{value}']?"
        )
    # Reject all non-iterables cleanly
    raise TypeError(
        f"In module '{module_name}': the '{field_name}' field must be a list, "
        f"but got {type(value).__name__}."
    )

def _ensure_string(value, field_name, module_name):
    if isinstance(value, str):
        return value
    else:
        raise TypeError(
            f"In module '{module_name}': the '{field_name}' field must be a string, "
            f"but got {type(value).__name__}."
        )

## TODO -- maybe don't do this at all? Metadata is optional
def _ensure_dict(value, field_name, module_name):
    if isinstance(value, dict):
        return value
    elif value is None:
        return value
    else:
        raise TypeError(
            f"In module '{module_name}': the '{field_name}' field must be a string, "
            f"but got {type(value).__name__}."
        )
        

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
        # Validate fields immediately ## TODO -- this is bare minimum. maybe do more? 
        validated_name = _ensure_string(name, "name", name)
        validated_produces = _ensure_list(produces, "produces", name)
        validated_requires = _ensure_list(requires, "requires", name)
        validated_metadata = _ensure_dict(metadata, "metadata", name)

        function.module_name = validated_name
        function.requires = tuple(validated_requires)
        function.produces = tuple(validated_produces)
        function.metadata = validated_metadata  # store additional metadata
        _modules[name] = function

        # Update derived caches
        for output in function.produces:
            _provides[output].append(function)
        for input_ in function.requires:
            _consumes[input_].append(function)

        return function
    return decorator

## TODO -- maybe don't print traceback by default?
def import_module_safely(modname: str):
    try:
        return importlib.import_module(modname)
    except ModuleNotFoundError as e:
        if e.name == modname:
            # True "module not found" error — clean
            logger.warning(
                "Failed to import module %r because it does not exist. "
                "Check PYTHONPATH or installation.",
                modname,
            )
        else:
            # Nested import failed inside the module
            tb = traceback.format_exc()
            logger.warning(
                "Module %r was found but failed during import due to a missing dependency: %s\n%s",
                modname,
                e,
                tb,
            )
        return None
    except Exception:
        tb = traceback.format_exc()
        logger.warning(
            "Module %r failed during import:\n%s",
            modname,
            tb,
        )
        return None

## TODO -- this should maybe return messages or something. make sure this is where we want it
def load_all_modules():
    """
    Automatically import all Python modules in target folder (flimsy.modules by default).
    Decorator-based registration ensures that each module is added to the registry.
    """
    imported = []
    modules_path = Path(flimsy.modules.__file__).parent
    logger.debug(modules_path)
    for loader, name, is_pkg in pkgutil.iter_modules([str(modules_path)]):
        try:
            _ = import_module_safely(f"flimsy.modules.{name}")
            imported.append(name)
        except Exception as e:
            logger.warning(f"Failed to import module '{name}': {e}")
    logger.info(f"Imported modules: {imported}")

def get_module(name: str) -> Callable:
    """Retrieve a registered module by name."""
    try:
        return _modules[name]
    except KeyError:
        raise KeyError(f"Module '{name}' not found. For list of available modules run `registry.list_modules()`")

def list_modules() -> List[str]:
    return list(_modules.keys())
    