import os
from time import time
import logging
import argparse

from flimsy.pipeline import logging_utils
from flimsy.utils.ioer import load_yaml, pretty_print
from flimsy.pipeline.pipeline import parse_config, validate_pipeline, run_pipeline

logger = logging.getLogger(__name__)

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', required=True) ## TODO -- finalize api
    
    parser.add_argument('--logdir', required=False)

    parser.add_argument('--overwrite', required=False, default='false', type=str, choices=['true', 'false', 'diff'])

    parser.add_argument('--loglevel', default='info', choices=['debug', 'info', 'warning', 'error', 'critical'],
            help='Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')

    return parser

def main():
    t1 = time()
    parser = get_parser()
    args = parser.parse_args()

    
    #Path(args.logdir, 'logs/').mkdir(parents=True, exist_ok=True) ## TODO -- maybe do this in logging? 
    logging_utils.configure_logging(os.path.join(args.logdir, 'logs/'), args.loglevel)
    logger.info('Welcome to flimsy - the Felsen Lab Integrated Modular analYsis package.')
    logger.debug(f'args:\n{args}')

    config = load_yaml(args.config) ## TODO -- this needs some reworking (see pipeline.py)
    logger.debug(config)
    pipeline = parse_config(config) ## TODO -- placeholder, since pipeline API not establishe yet. Probably need to return list of modules + some kind of param store for actually running it
    validate_pipeline(pipeline) ## TODO -- may need to pass params from config or something here
    logger.debug('Pipeline validated')
    logger.debug('Pipeline execution started')
    summary = run_pipeline(pipeline) ## TODO -- may need to pass params from config or something here
    logger.debug('Pipeline execution complete')
    
    t2 = time()
    
    logger.info(pretty_print(summary))
    logger.info(f'Execution completed (runtime={t2-t1})')

if __name__ == "__main__":
    main()