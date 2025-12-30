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
from flimsy.pipeline.basemodule import validate_output


logger = logging.getLogger(__name__)

## TODO -- not sure if this is required. should be able to directly access modules list in run_pipeline, likewise with params if present. more urgently needed is the run config. do we even want to allow separating them?
#      longer term, we may want to separate parsing from execution --> maybe we want to do get_module in parsing and then we can do param unpacking in run?
def parse_config(config):
    """Parses pipeline configs"""
    logger.debug(config)
    return []
    
def validate_pipeline(pipeline):
    """Checks that provided pipeline config is valid (all inputs and dependencies are satisfied, all modules exist)"""
    logger.warning('Not implemented')

## TODO -- directly taking config for now
def run_pipeline(pipeline):
    """"""
    summary = {}
    load_all_modules()
    logger.debug(list_modules())

    ## TODO -- where does file come from? Do we want to require the first module to be a file creation module? How do we handle nwb vs h5 etc
    logger.debug(pipeline)
    for module_name in pipeline:
        module = get_module(module_name)
        logger.debug(module)
        output, messages = module(params[module_name])
    
        summary[module_name] = messages    
        validate_output(module_name, output) #Checks that all expected keys are present and checks for common failure modes (nans, infs, etc)
        ## TODO -- check that all declared fields were produced
        ## TODO -- check that no undeclared fields were produced ({fields_before} - {fields_after} == {module_produces})
        ## TODO -- run validation on declared fields

        #store output
    
    return summary