# def compute_frame_timestamps_for_crystals_sessions(
#     home_folder,
#     side='right', ## We only need to extract timestamps for one side, since the other side will be aligned to the same timestamps. 
#     lag=-4.37 ## 4.37 seconds is the MEAN time between clicking "start" and the first stimulus appearing on screen. We subtract this from the first stimulus timestamp to generate frametimestamps
#     ):
#     """
#     """

#     home_folder = pl.Path(home_folder)
#     timestamp_files = list(home_folder.joinpath('videos').rglob('*_timestamps.txt'))

#     frameTimestamps = None
#     for f in timestamp_files:
#         if side in f.name.lower():

#             # Load inter-frame intervals for the target camera
#             ifi = np.loadtxt(f)
#             frameTimestamps = np.concatenate([
#                 np.array([0]),
#                 np.cumsum(ifi[1:]) / 1000000000
#             ])

#             # Identify the first visual stimulus timestamp
#             t0 = None
#             for f in home_folder.joinpath('videos').iterdir():
#                 if f.stem.lower() == 'driftinggratingmetadata':
#                     with open(f, 'r') as stream:
#                         lines = stream.readlines()[5:]
#                     a, b, c, d, e = lines[0].rstrip('\n').split(', ')
#                     t0 = float(e)
#             if t0 is None:
#                 raise Exception('Could not identify timestamp for the first visual event')
            
#             # Add the timestamp of the first visual event + the constant lag
#             frameTimestamps = frameTimestamps + t0 + lag

#     #
#     if frameTimestamps is None:
#         raise Exception('Could not compute frame timestamps')

#     return frameTimestamps


import logging

import numpy as np
import pandas as pd

from flimsy.pipeline.basemodule import *
from flimsy.utils.signal_processing import find_edges
from flimsy.utils.ioer import find_files_matching_pattern

@module(name="timestampCameraFramesNoLabjack", description="")

@requires("metadata/basepath")
@produces("labjack/cameras/timestamps", description="Labjack sample where frame events occur")
@produces("frames/{side}/intervals", description="")
# @produces("frames/left/intervals", description="")
# @produces("frames/right/intervals", description="")
@param("sampling_rate", description="labjack sampling rate")
@param("interval_pattern")
@param("start_offset", default=4.37, description="4.37 seconds is the MEAN time between clicking 'start' and the first stimulus appearing on screen")
@param("fieldnames")
def run(data, params):

    logger.debug(f"params: {params.keys()}")

    
    sampling_rate = params["sampling_rate"]
    basepath = data["metadata/basepath"]
    interval_pattern = params["interval_pattern"]

    res = {}

    n_frames = 0
    interval_files = find_files_matching_pattern(basepath, interval_pattern, recursive=True)
    for side in params["fieldnames"]["side"]:
        for filename in interval_files:
            if side in filename.name:
                interval_runner = np.loadtxt(filename)
                res[f"frames/{side}/intervals"] = interval_runner
                frame_runner = len(interval_runner)
                if frame_runner > n_frames:
                    n_frames = frame_runner

    epsilon = 0.1 ## floating point rounding correction
    timestamps = np.arange(0, (n_frames+1)*(1/sampling_rate)+epsilon, 1/sampling_rate) + lag
    res["labjack/cameras/timestamps"] = timestamps

    return res