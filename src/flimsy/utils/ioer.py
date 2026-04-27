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

import re
import io
import pandas as pd
 
def parse_experiment_file(filepath: str) -> tuple[dict, pd.DataFrame]:
    """
    Parse a experiment .txt file with a metadata header and tabular data.
 
    The file format is:
      - Lines 1–4: key-value metadata (e.g. "Spatial frequency: 0.15 (cycles/degree)")
      - Line 5:    column names prefixed with "Columns: ", may contain unprotected commas
                   inside parenthetical descriptions
      - Line 6+:   numeric CSV data
 
    Returns
    -------
    metadata : dict
        Keys are the parameter names (str), values are the raw value strings (str).
    df : pd.DataFrame
        Tabular data with cleaned column names.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()
 
    # ── 1. Metadata (first 4 lines) ──────────────────────────────────────────
    metadata = {}
    METADATA_LINES = 4
    for line in lines[:METADATA_LINES]:
        # Split only on the first colon so values like "0.4 (0, 1)" stay intact
        key, _, value = line.partition(":")
        metadata[key.strip()] = value.strip()
 
    # ── 2. Column names (line 5) ─────────────────────────────────────────────
    # Strip the "Columns: " prefix, then split on commas that are NOT inside
    # parentheses — this preserves descriptions like "Event (1=Grating, 2=Motion)"
    columns_line = lines[METADATA_LINES].strip()
    columns_line = re.sub(r"^Columns:\s*", "", columns_line, flags=re.IGNORECASE)
 
    # Split on commas that have no open parenthesis to their left (within the token)
    column_names = [c.strip() for c in re.split(r",(?![^(]*\))", columns_line)]
 
    # ── 3. Tabular data (remaining lines) ────────────────────────────────────
    data_block = "".join(lines[METADATA_LINES + 1 :])
    df = pd.read_csv(
        io.StringIO(data_block),
        header=None,
        names=column_names,
        skipinitialspace=True,
    )
 
    return metadata, df
