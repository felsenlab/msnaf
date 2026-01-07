import os

import h5py

from flimsy.pipeline.basemodule import *
from flimsy.utils.ioer import load_yaml, find_files_matching_pattern

logger = logging.getLogger(__name__)

@module(name="initH5", description="")
@param("basepath", description="Folder that contains session data. Also serves as the output directory.")
@param("prefix", default="", description="String that gets pre-pended to the output file name. For example, if provided, the output file will be named {prefix}_results.h5")
@param("metadata_pattern")
@produces("metadata/basepath", description="Saves location of the session folder for access by other modules")
def run(basepath, prefix, metadata_pattern):
    ## TODO -- figure out how this is being passed so we can 
    # make sure to append results.h5 to it or something

    file = h5py.File(os.path.join(basepath, prefix, 'results.h5'), "w")

    filepath = find_files_matching_pattern(basepath, metadata_pattern)
    if len(filepath) != 1:
        logger.error(f"Exactly one metdata file is required, got {filepath}")
    session_info = load_yaml(filepath.pop())

    return {'metadata/basepath':basepath, "metadata/session_info":session_info}, file