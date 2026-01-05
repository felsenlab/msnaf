import logging

from flimsy.pipeline.basemodule import *
from flimsy.utils.signal_processing import find_edges, compute_threshold

@module(name="", description="")
@requires()
@produces()
@param()
def run():
    pass


## TODO -- name better
def identify_dropped_frames():
    pass


    

## TODO -- unify into one detect_edges function that has an optional filter boolean?
def parse_camera_signal(camera_signal):
    """
    Processes camera clock signal. This signal represents an attempt to retrieve
    a camera frame, but does not indicate success or failure. A separate script is
    needed to identify dropped frames. The signal is typically very clean and does 
    not require filtering.
    """
    binarized = camera_signal > camera_signal.mean()
    edge_indices = np.where(np.diff(binarized)==1)[0]
    
    return edge_indices
