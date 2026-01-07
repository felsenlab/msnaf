import logging

import numpy as np

from flimsy.pipeline.basemodule import *

logger = logging.getLogger(__name__)

@module(name="timestampSaccades", description="")
#@requires("saccades/predicted/{side}/epochs")
@requires("saccades/predicted/left/epochs")
@requires("saccades/predicted/right/epochs")
#@requires("frames/{side}/timestamps") ## TODO -- need to resolve this 
@requires("frames/left/timestamps")
@requires("frames/righ/timestamps")
#@produces("saccades/predicted/{side}/timestamps", description="")
@produces("saccades/predicted/left/timestamps", description="")
@produces("saccades/predicted/right/timestamps", description="")
def run(data, params):
    left_frame_timestamps = data["frames/left/timestamps"]
    right_frame_timestamps = data["frames/right/timestamps"]

    left_saccade_epochs = data["saccades/predicted/left/epochs"]
    right_saccade_epochs = data["saccades/predicted/right/epochs"]

    #left_saccade_timestamps = left_frame_timestamps[left_saccade_epochs] ## TODO -- fractional frames with interp
    #right_saccade_timestamps = right_frame_timestamps[right_saccade_epochs]
    left_saccade_timestamps = np.interp(left_saccade_epochs, range(0, len(left_frame_timestamps), left_frame_timestamps))
    right_saccade_timestamps = np.interp(right_saccade_epochs, range(0, len(right_frame_timestamps), right_frame_timestamps))

    return {"saccades/predicted/left/timestamps":left_saccade_timestamps, "saccades/predicted/right/timestamps":right_saccade_timestamps}

# def run(saccade_epochs=None, frame_timestamps=None):
#     saccade_timestamps = frame_timestamps[saccade_epochs]

#     return saccade_timestamps