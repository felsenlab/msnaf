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

logger = logging.getLogger(__name__)


def parse_pipeline(config):
    """Parses pipeline configs"""
    
    

def validate_pipeline():
    """Checks that provided pipeline config is valid (all inputs and dependencies are satisfied, all modules exist)"""
    pass

def run_pipeline():
    """"""
    pass