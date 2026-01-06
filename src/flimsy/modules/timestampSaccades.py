import logging

import numpy as np

from flimsy.pipeline.basemodule import *

logger = logging.getLogger(__name__)

@module(name="", description="")
@requires("saccades/predicted/{side}/epochs")
@requires("frames/{side}/timestamps") ## TODO -- should we make this static? 
@produces("saccades/predicted/{side}/timestamps", description="")
def run(saccade_onsets, saccade_offsets, timestamps):
    onset_timestamps = timestamps[saccade_onsets]
    offset_timestamps = timestamps[saccade_offsets]

    return onset_timestamps, offset_timestamps