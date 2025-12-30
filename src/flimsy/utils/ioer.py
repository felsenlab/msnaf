import yaml
import logging
import importlib
import traceback
import pickle as pkl
from pathlib import Path

logger = logging.getLogger(__name__)

def load_yaml(path: str) -> dict:
    """
    Load a YAML file and return the parsed Python object.
    
    Parameters
    ----------
    path : str or Path
        Path to the YAML file.
    
    Returns
    -------
    Any
        Parsed content of the YAML file.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_pickle(path):
    """
    Load a pickle file and return the deserialized Python object.

    Parameters
    ----------
    path : str or Path
        Path to the pickle file.

    Returns
    -------
    Any
        The unpickled Python object.
    """
    path = Path(path)
    with path.open("rb") as f:
        return pkl.load(f)


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


def find_files_matching_pattern(path, pattern, recursive=False):
    root = Path(path)
    files = root.rglob(pattern) if recursive else root.glob(pattern)
    return list(files)


def pretty_print(summary): ## TOOD -- should this be generalized or only for run summaries? Should this be here or do we need a logging module?
    return ''