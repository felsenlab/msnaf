"""
The registry is a central respository of available modules.
"""
import pkgutil
import logging
import importlib
from pathlib import Path
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import flimsy.modules

logger = logging.getLogger(__name__)

_MODULES: Dict[str, Dict[str, Any]] = {}
_VALIDATORS: Dict[str, Dict[str, Any]] = {}
_MODULES_BY_NAMESPACE: Dict[str, set] = defaultdict(set)

_MODULES_LOADED = False


############ HELPER FUNCTIONS ############
def _append_attr(fn: Callable, attr: str, value: dict):
    if not hasattr(fn, attr):
        setattr(fn, attr, [])
    getattr(fn, attr).append(value)

def _get_attr_list(fn: Callable, attr: str) -> List[dict]:
    return getattr(fn, attr, [])

def _ensure_loaded():
    global _MODULES_LOADED
    if not _MODULES_LOADED:
        load_all_modules()
        _MODULES_LOADED = True

## TODO -- move to utils.ioer
## TODO -- integrate with messaging system
def load_all_modules(package: str = "flimsy.modules"):
    try:
        pkg = importlib.import_module(package)
    except ImportError:
        logger.warning(f"Module package '{package}' not found")
        return

    if not hasattr(pkg, "__path__"):
        return

    imported = []
    for _, name, _ in pkgutil.iter_modules(pkg.__path__):
        fqname = f"{package}.{name}"
        try:
            importlib.import_module(fqname)
            imported.append(name)
        except Exception:
            logger.exception(f"Failed to import module '{fqname}'")

    logger.debug(f"Imported modules: {imported}")


############ DECORATORS ############
def requires(
    name: str,
    *,
    dtype: Optional[str] = None,
    optional: bool = False,
    description: Optional[str] = None,
):
    def decorator(fn: Callable):
        _append_attr(
            fn,
            "_flimsy_requires",
            {
                "name": name,
                "dtype": dtype,
                "optional": optional,
                "description": description,
            },
        )
        return fn
    return decorator

def produces(
    name: str,
    *,
    dtype: Optional[str] = None,
    validators: Optional[List[str]] = None,
    description: Optional[str] = None,
):
    def decorator(fn: Callable):
        if validators is not None and not isinstance(validators, list):
            raise TypeError(
                f"validators for output '{name}' must be a list of validator names"
            )
        _append_attr(
            fn,
            "_flimsy_produces",
            {
                "name": name,
                "dtype": dtype,
                "validators": validators or [],
                "description": description,
            },
        )
        return fn
    return decorator

def param(name: str, dtype: Any = None, default: Any = None, description: str = None):
    """Declare a parameter/argument that tweaks module behavior."""
    def decorator(fn):
        _append_attr(fn, "_flimsy_params", {
            "name": name, "dtype": dtype, "default": default, "description": description
        })
        return fn
    return decorator

## TODO -- integrate with message system
def module(*, name: str, description: Optional[str]=None, namespace: Optional[str]=None, tags: Optional[List[str]]=None):
    def decorator(fn: Callable):
        if name in _MODULES:
            raise ValueError(f"Module '{name}' already registered") 

        requires_meta = _get_attr_list(fn, "_flimsy_requires")
        produces_meta = _get_attr_list(fn, "_flimsy_produces")
        params_meta = getattr(fn, "_flimsy_params", [])

        if not produces_meta:
            raise ValueError(f"Module '{name}' must declare at least one @produces")

        ns = namespace or "default"
        _MODULES_BY_NAMESPACE[ns].add(name)
        _MODULES[name] = {
            "name": name,
            "fn": fn,
            "description": description or "",
            "namespace": ns,
            "tags": tags or [],
            "requires": requires_meta,
            "produces": produces_meta,
            "params": params_meta,
        }

        return fn
    return decorator

def validator(name: str):
    def decorator(fn: Callable):
        if name in _VALIDATORS:
            raise ValueError(f"Validator '{name}' already registered")
        _VALIDATORS[name] = fn
        return fn
    return decorator

## TODO -- add param decorator

############ MODULE VALIDATION #####################
## TODO -- needs checking
def validate_module_registry():
    """Check all registered modules for common errors."""
    errors = []
    for name, meta in _MODULES.items():
        # Check requires/produces lists are present
        if not isinstance(meta.get("requires", []), list):
            errors.append(f"Module '{name}' requires field is not a list")
        if not isinstance(meta.get("produces", []), list):
            errors.append(f"Module '{name}' produces field is not a list")
        # Check params
        for p in meta.get("params", []):
            if "name" not in p:
                errors.append(f"Module '{name}' has a param missing 'name'")
        # Optional: check for name collisions between requires/produces/params
        requires_names = {r["name"] for r in meta.get("requires", [])}
        produces_names = {p["name"] for p in meta.get("produces", [])}
        params_names = {p["name"] for p in meta.get("params", [])}
        intersection = (requires_names & produces_names) | (requires_names & params_names) | (produces_names & params_names)
        if intersection:
            errors.append(f"Module '{name}' has overlapping names across requires/produces/params: {intersection}")
    return errors

############ PUBLIC GETTERS ############


def summarize_module(module_name: str, show_paths: bool = True) -> str:
    """
    Generate a formatted summary of a module.

    Args:
        module: A module object registered with @module decorator.
        show_paths: If True, display the HDF5 paths for requires/produces.

    Returns:
        Formatted string summary of the module.
    """
    if not _MODULES_LOADED:
        load_all_modules()

    lines = []

    # Basic info
    if module_name in _MODULES:
        module = get_module(module_name)
    else:
        logger.warning(f"{module_name} is not a registered module. Use --list_modules for a list of all registered modules")
        return
    description = module['description']
    lines.append(f"Module: {module_name}")
    lines.append(f"Description: {description}\n")

    # Parameters
    params = module["params"]
    if params:
        lines.append("Parameters:")
        for p in params:
            pname = p.get("name", "UNKNOWN")
            pdesc = p.get("description", "")
            pdefault = p.get("default", "<required>")
            lines.append(f"  - {pname}: {pdesc} (default: {pdefault})")
        lines.append("")

    # Requires
    requires = module["requires"]
    if requires:
        lines.append("Required inputs:")
        for r in requires:
            rname = r.get("name", "UNKNOWN")
            rdesc = r.get("description", "")
            rpath = r.get("path", "")
            if show_paths:
                lines.append(f"  - {rname}: {rdesc} (path: {rpath})")
            else:
                lines.append(f"  - {rname}: {rdesc}")
        lines.append("")

    # Produces
    produces = module["produces"]
    if produces:
        lines.append("Outputs:")
        for p in produces:
            pname = p.get("name", "UNKNOWN")
            pdesc = p.get("description", "")
            ppath = p.get("path", "")
            if show_paths:
                lines.append(f"  - {pname}: {pdesc} (path: {ppath})")
            else:
                lines.append(f"  - {pname}: {pdesc}")
        lines.append("")

    return "\n".join(lines)

## TODO -- these are all wrong. ensure they are up to date with the API (specifically, need to handle namespaces)
def list_modules() -> List[str]:
    _ensure_loaded()
    return list(_MODULES.keys())


def get_module(name: str) -> Dict[str, Any]:
    _ensure_loaded()
    return _MODULES[name]


def list_outputs() -> List[str]:
    _ensure_loaded()
    out = set()
    for m in _MODULES.values():
        for p in m["produces"]:
            out.add(p["name"])
    return sorted(out)


def find_producers(field: str) -> List[str]:
    _ensure_loaded()
    return [
        name
        for name, m in _MODULES.items()
        if any(p["name"] == field for p in m["produces"])
    ]


def get_validator(name: str) -> Callable:
    return _VALIDATORS[name]


def list_validators() -> List[str]:
    return sorted(_VALIDATORS.keys())

### OLD
# _modules = {}
# _provides = defaultdict(list)
# _consumes = defaultdict(list)


        
## TODO -- add validation tag --> allow_nans, is_numeric, etc that allows user to select automated validation
# def module(name: str, requires: Tuple[str, ...] = (), produces: Tuple[str, ...] = (), metadata: Dict=None):
#     """
#     Decorator to register a function as a pipeline module.

#     Parameters:
#         name: module name
#         requires: tuple of required input fields
#         produces: tuple of output fields
#         metadata: optional extra metadata for future use
#     """
#     def decorator(function: Callable):
#         # Validate fields immediately ## TODO -- this is bare minimum. maybe do more? 
#         validated_name = _ensure_string(name, "name", name)
#         validated_produces = _ensure_list(produces, "produces", name)
#         validated_requires = _ensure_list(requires, "requires", name)
#         validated_metadata = _ensure_dict(metadata, "metadata", name)

#         function.module_name = validated_name
#         function.requires = tuple(validated_requires)
#         function.produces = tuple(validated_produces)
#         function.metadata = validated_metadata  # store additional metadata
#         _modules[name] = function

#         # Update derived caches
#         for output in function.produces:
#             _provides[output].append(name)
#         for input_ in function.requires:
#             _consumes[input_].append(name)

#         return function
#     return decorator



# ## TODO -- this should maybe return messages or something. make sure this is where we want it
# def load_all_modules():
#     """
#     Automatically import all Python modules in target folder (flimsy.modules by default).
#     Decorator-based registration ensures that each module is added to the registry.
#     """
#     imported = []
#     modules_path = Path(flimsy.modules.__file__).parent
#     logger.debug(modules_path)
#     for loader, name, is_pkg in pkgutil.iter_modules([str(modules_path)]):
#         try:
#             _ = import_module_safely(f"flimsy.modules.{name}")
#             imported.append(name)
#         except Exception as e:
#             logger.warning(f"Failed to import module '{name}': {e}")
#     logger.info(f"Imported modules: {imported}")

# def get_module(name: str) -> Callable:
#     """Retrieve a registered module by name."""
#     try:
#         return _modules[name]
#     except KeyError:
#         raise KeyError(f"Module '{name}' not found. For list of available modules run `registry.list_modules()`")

# def list_modules() -> List[str]:
#     return list(_modules.keys())
    