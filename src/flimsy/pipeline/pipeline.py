# Pipelines are collections of modules that are run sequentially. 
# Pipelines are defined by config files which specify execution
# order and default params. Pipelines are executed by providing
# a run config, which provides required inputs (such as files)
# and the desired params for the run. Run configs support any 
# number of inputs (e.g., you can run pipelines on one file or
# a hundred). Params can be specified per file or per run. Note
# that these params override the defaults in the pipeline config
# files.
# 
# Examples of both of these files are provided in flimsy/examples


import logging

import flimsy.pipeline.registry
from flimsy.pipeline.registry import get_module, list_modules, load_all_modules
from flimsy.pipeline.base import validate_output


logger = logging.getLogger(__name__)


def parse_config(config):
    """Parses pipeline configs"""
    return []
    
def validate_pipeline(pipeline):
    """Checks that provided pipeline config is valid (all inputs and dependencies are satisfied, all modules exist)"""
    pass

def run_pipeline(pipeline):
    """"""
    summary = {}
    load_all_modules()
    logger.debug(list_modules())

    ## TODO -- where does file come from? Do we want to require the first module to be a file creation module? How do we handle nwb vs h5 etc
    for module_name in pipeline:
        module = get_module(module_name)
        output, messages = module(params[module_name])
    
        summary[module_name] = messages    
        validate_output(module_name, output) #Checks that all expected keys are present and checks for common failure modes (nans, infs, etc)

        #store output
    
    return summary