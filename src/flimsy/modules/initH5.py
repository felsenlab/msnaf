import os

import h5py

from flimsy.pipeline.basemodule import *

logger = logging.getLogger(__name__)

## TODO -- is this necessary or should we make this part of the pipeline 
#          (we need some way to make switching the output easy)
@module(name="init_h5_file", description="")
@requires("filepath")
@param("prefix", default='')
@produces("dummy") ## TODO -- how do we handle modules that don't produce h5 fields? Since they *should* all produce files, maybe this is trivial? We can just declare the path and check that the path exists
def run(filepath, prefix):
	## TODO -- figure out how this is being passed so we can 
	# make sure to append results.h5 to it or something
	file = h5py.File(os.path.join(filepath, prefix), "w")
	file.close()