import logging
import os
from datetime import datetime
import colorlog

logger = logging.getLogger(__name__)

def loglevel_type(level_str):
    """Convert string to logging level constant."""
    try:
        return getattr(logging, level_str.upper())
    except AttributeError:
        raise argparse.ArgumentTypeError(f"Invalid log level: {level_str}")

def configure_logging(log_dir: str, level='info'):
    """Configure the root logger for the pipeline."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    current_time = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
    
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s" + log_format,
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'white,bg_red'
        }))

    if log_dir is not None:
        Path(log_dir).mkdir(exist_ok=True)
        log_file = os.path.join(log_dir, f'{current_time}.log')
        #Path(log_file).parent.mkdir(parents=True, exist_ok=True) ## TODO -- consider how we want to handle file creation

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))

    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])
    logger.info('Logging to %s' % log_file)