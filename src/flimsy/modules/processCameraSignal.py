import logging

import numpy as np

from flimsy.pipeline.basemodule import *
from flimsy.utils.signal_processing import find_edges, compute_threshold

@module(name="", description="")
@requires()
@produces("frame_indices", description="Labjack sample where frame events occur")
@param()
def run():
    pass


## TODO -- name better
def identify_dropped_frames():
    pass


    

## TODO -- can we make this more precise with intervals file?
## TODO -- should we identify dropped frames here so we don't have to deal with it during saccade extraction? --> correcting timestamps at that point is inconvenient
def parse_camera_signal(camera_signal, sampling_rate):
    """
    Processes camera clock signal. This signal represents an attempt to retrieve
    a camera frame, but does not indicate success or failure. A separate script is
    needed to identify dropped frames. Signal is digital.
    """
    _, _, edge_indices = find_edges(camera_signal, filter=False)
    timevalues = np.arange(0, np.ceil(len(camera_signal)/sampling_rate), 1/sampling_rate)

    timestamps = timevalues[edge_indices]
    
    return timestamps, edge_indices
