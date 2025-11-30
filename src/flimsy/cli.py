import os
import logging
import argparse
from pathlib import Path

from flimsy.pipeline import logging_utils

logger = logging.getLogger(__name__)

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir', required=False)

    parser.add_argument('--overwrite', required=False, default='false', type=str, choices=['true', 'false', 'diff'])

    parser.add_argument('--loglevel', default='info', choices=['debug', 'info', 'warning', 'error', 'critical'],
            help='Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    
    Path(args.logdir, 'logs/').mkdir(parents=True, exist_ok=True) ## TODO -- maybe do this in logging? 
    logging_utils.configure_logging(os.path.join(args.logdir, 'logs/'), args.loglevel)
    logger.info('Welcome to flimsy - the Felsen Lab Integrated Modular analYsis package.')
    logger.debug(f'args:\n{args}')

if __name__ == "__main__":
    main()