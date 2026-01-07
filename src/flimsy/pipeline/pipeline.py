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
from typing import List

from flimsy.pipeline.registry import get_module, load_all_modules
from flimsy.utils.ioer import load_datasets, save_to_h5
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

def resolve_module_params(module_dict, pipeline_defaults: dict, run_params: dict) -> dict:
    """
    Combine default params from @param decorator, pipeline config, and run config.
    Precedence (highest wins):
        1. run_params
        2. pipeline_defaults
        3. decorator defaults
    """

    # {
    #     "name": name,
    #     "fn": fn,
    #     "description": description or "",
    #     "namespace": ns,
    #     "tags": tags or [],
    #     "requires": requires_meta,
    #     "produces": produces_meta,
    #     "params": params_meta,
    # }

    resolved = {}
    name = module_dict["name"]

    try:
        for param in module_dict["params"]:
            #logger.warning(param)
            #logger.warning(module_dict["params"])
            default = param.get("default", None)
            if default is not None:
                resolved[param["name"]] = default

        for param in pipeline_defaults.get(name, []):
            default = pipeline_defaults[name].get(param, None)
            if default is not None:
                resolved[param] = default

        for param in run_params.get(name, []):
            default = run_params[name].get(param, None)
            if default is not None:
                resolved[param] = default
    except Exception as e:
        logger.error(name)
        logger.error(param)
        logger.error(param.get('default'))
        logger.error(e)


    # # 1. decorator defaults
    # for param_def in getattr(module_fn, "_params", []):
    #     name = param_def["name"]
    #     default = param_def.get("default")
    #     resolved[name] = default

    # # 2. pipeline-wide defaults
    # resolved.update(pipeline_defaults.get(module_fn._module_name, {}))

    # # 3. run-time params
    # resolved.update(run_params.get(module_fn._module_name, {}))

    return resolved

## TODO -- directly taking config for now

## TODO -- figure out the argument passing. options are datasets, params or name everything --> runs into issues with dynamism. OR OR OR OR we can just hard code for now
## TODO -- saccade timestamping needs fractional frames
## TODO -- modules like saccade timestamping and extract need looping to handle sides OR need to be run multiple times
## TODO -- all of this needs to be tested

def run_pipeline(pipeline, run_config):
    """
    """
    summary = {}
    load_all_modules()
    #logger.debug(list_modules())

    ## TODO -- where does file come from? Do we want to require the first module to be a file creation module? How do we handle nwb vs h5 etc
    #logger.debug(pipeline)

    initfile = get_module("initH5")

    #params = resolve_module_params(initfile, pipeline, run_config)
    #logger.error(run_config["initH5"])
    basepath_list: List[str] = run_config["initH5"]["basepath"]
    init_params = pipeline.pop("initH5")
    initfn = initfile["fn"]
    for basepath in basepath_list:
        #datasets = load_datasets(initfile.requires)
        #logger.error(basepath)
        #logger.error(basepath_list)
        output, file = initfn(basepath, "", init_params["metadata_pattern"])
        save_to_h5(file, output)
        
        for module_name in pipeline:
            logger.debug(module_name)
            module = get_module(module_name)
            fn = module["fn"]

            params = resolve_module_params(module, pipeline, run_config)
            #logger.error(module["requires"])
            datasets = load_datasets(file, [requires_list["name"] for requires_list in module["requires"]])

            output = fn(datasets, params)

            save_to_h5(file, output)

            logger.error(file.keys())
    
            #summary[module_name] = messages    
            #validate_output(module_name, output) #Checks that all expected keys are present and checks for common failure modes (nans, infs, etc)
            ## TODO -- check that all declared fields were produced
            ## TODO -- check that no undeclared fields were produced ({fields_before} - {fields_after} == {module_produces})
            ## TODO -- run validation on declared fields

            #store output
    
    return summary