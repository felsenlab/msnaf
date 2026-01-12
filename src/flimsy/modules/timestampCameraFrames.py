import logging

import numpy as np
import pandas as pd

from flimsy.pipeline.basemodule import *
from flimsy.utils.signal_processing import find_edges
from flimsy.utils.ioer import find_files_matching_pattern

@module(name="timestampCameraFrames", description="")
@requires("labjack/camera/raw")
@requires("metadata/basepath")
@produces("labjack/cameras/timestamps", description="Labjack sample where frame events occur")
@produces("frames/left/intervals", description="")
@produces("frames/right/intervals", description="")
@param("sampling_rate", description="labjack sampling rate")
@param("interval_pattern")
def run(data, params):
    camera_signal = data["labjack/camera/raw"]
    sampling_rate = params["sampling_rate"]

    basepath = data["metadata/basepath"]
    interval_pattern = params["interval_pattern"]

    timestamps, edge_indices = parse_camera_signal(camera_signal, sampling_rate)

    res = {}

    interval_files = find_files_matching_pattern(basepath, interval_pattern, recursive=True)
    for side in ["left", "right"]:
        for filename in interval_files:
            if side in filename.name:
                res[f"frames/{side}/intervals"] = pd.read_csv(filename).to_numpy()

    res["labjack/cameras/timestamps"] = timestamps

    return res


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
