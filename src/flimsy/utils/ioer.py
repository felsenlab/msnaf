import yaml
import logging
import importlib
import traceback
import pickle as pkl
from pathlib import Path
from typing import Dict

import numpy as np


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

def load_datasets(h5file, fieldnames):
    to_return = {}
    for name in fieldnames:
        logger.debug(f"loading: {name}")
        to_return[name] = read_obj(h5file[name])
    
    return to_return

def save_to_h5(h5file, data: Dict):
    for field_name, field_data in data.items():
        #logger.error(type(field_data))
        logger.debug(f"Saving {field_name}")
        #h5file.create_dataset(field_name)
        write_obj(h5file, field_name, field_data)
        #h5file[field_name] = field_data

def read_obj(h5obj):
    """
    Convert an HDF5 object (group or dataset) into a native Python object.
    """
    obj_type = h5obj.attrs.get("_type")

    if obj_type == "dict":
        return {k: read_obj(v) for k, v in h5obj.items()}

    if obj_type == "ndarray":
        return np.array(h5obj, copy=False)

    if obj_type == "scalar":
        return h5obj[()]

    if obj_type == "str":
        val = h5obj[()]
        return val.decode() if isinstance(val, (bytes, np.bytes_)) else str(val)
    
    logger.error(f"Unsupported type: {type(h5obj)}")

def write_obj(h5file, key, value):
    """
    Write a Python object to an HDF5 group.
    """
    if isinstance(value, dict):
        grp = h5file.create_group(key)
        grp.attrs["_type"] = "dict"
        for k, v in value.items():
            write_obj(grp, str(k), v)

    elif isinstance(value, np.ndarray):
        dset = h5file.create_dataset(key, data=value)
        dset.attrs["_type"] = "ndarray"

    elif isinstance(value, (int, float, np.integer, np.floating)):
        dset = h5file.create_dataset(key, data=value)
        dset.attrs["_type"] = "scalar"

    elif isinstance(value, str):
        dset = h5file.create_dataset(
            key, data=np.string_(value)
        )
        dset.attrs["_type"] = "str"

    else:
        logger.error(f"Unsupported type: {type(value)}")