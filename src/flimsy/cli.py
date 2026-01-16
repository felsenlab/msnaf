import os
import sys
import logging
import argparse
from time import time


from flimsy.pipeline import logging_utils
from flimsy.utils.ioer import load_yaml, pretty_print
from flimsy.pipeline.pipeline import parse_config, validate_pipeline, run_pipeline
from flimsy.pipeline.registry import summarize_module

logger = logging.getLogger(__name__)

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pipeline') ## TODO -- finalize api
    parser.add_argument('--run') ## TODO -- finalize api
    
    parser.add_argument('--logdir', required=False, default=".")

    parser.add_argument('--module_help', type=str)

    parser.add_argument('--overwrite', required=False, default='false', type=str, choices=['true', 'false', 'diff'])

    parser.add_argument('--loglevel', default='DEBUG', choices=['debug', 'info', 'warning', 'error', 'critical'],
            help='Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')

    return parser

def print_module_summary(module_name: str, show_paths: bool = True):
    """
    Convenience function to print the module summary to stdout.
    """
    summary = summarize_module(module_name, show_paths=show_paths)
    print(summary)
    sys.exit(0)

def main():
    t1 = time()
    parser = get_parser()
    args = parser.parse_args()
    
    #Path(args.logdir, 'logs/').mkdir(parents=True, exist_ok=True) ## TODO -- maybe do this in logging? 
    logging_utils.configure_logging(os.path.join(args.logdir, 'logs/'), args.loglevel)
    logger.info('Welcome to flimsy - the Felsen Lab Integrated Modular analYsis package.')
    logger.debug(f'args:\n{args}')

    if args.module_help:
        print_module_summary(args.module_help)

    if args.pipeline is None or args.run is None:
        logger.error("--pipeline and --run MUST be provided")

    pipeline_config = load_yaml(args.pipeline) ## TODO -- this needs some reworking (see pipeline.py)
    run_config = load_yaml(args.run)
    #logger.debug(config)
    #pipeline = parse_config(config) ## TODO -- placeholder, since pipeline API not establishe yet. Probably need to return list of modules + some kind of param store for actually running it
    #validate_pipeline(pipeline) ## TODO -- may need to pass params from config or something here
    #logger.debug('Pipeline validated')
    #logger.debug('Pipeline execution started')
    summary = run_pipeline(pipeline_config, run_config) ## TODO -- may need to pass params from config or something here
    logger.debug('Pipeline execution complete')
    
    t2 = time()
    
    logger.info(pretty_print(summary))
    logger.info(f'Execution completed (runtime={t2-t1:.3f})')

if __name__ == "__main__":
    main()