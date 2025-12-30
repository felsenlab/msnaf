import logging

import pandas as pd

from flimsy.pipeline.registry import module

logger = logging.getLogger(__name__)

@module(name='parseStimulusMetadata', requires=['metadata_filepath'])
def run():
	pass