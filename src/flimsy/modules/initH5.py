import os

import h5py

from flimsy.pipeline.basemodule import *

logger = logging.getLogger(__name__)

@module(name="init_h5", description="")
@param("basepath", description="Folder that contains session data. Also serves as the output directory.")
@param("prefix", default="", desrciption="String that gets pre-pended to the output file name. For example, if provided, the output file will be named {prefix}_results.h5")
@produces("metadata/basepath", description="Saves location of the session folder for access by other modules")
def run(basepath, prefix):
	## TODO -- figure out how this is being passed so we can 
	# make sure to append results.h5 to it or something
	file = h5py.File(os.path.join(basepath, prefix), "w")
	file.close()