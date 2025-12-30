import logging

from flimsy.pipeline.basemodule import *
from flimsy.utils.signal_processing import find_edges, compute_threshold

@module()
@requires()
@produces()
def run():
    pass


### Scratch

## TODO -- this should warn about poorly separated means
## TODO -- this should warn when not bimodal
## TODO -- this should wran about
def get_state_means(signal):
    gmm = GaussianMixture(n_components=2, covariance_type='full')
    gmm.fit(signal.reshape(-1,1))

    means = gmm.means_.ravel()
    stds  = np.sqrt(gmm.covariances_).ravel()

    return means, stds

def compute_threshold(state_means, state_stdevs, mode):
    if mode == 'bayes':
        return compute_bayes_threshold()
    elif mode == 'mean':
        return state_means.sum()/2

from scipy.ndimage import median_filter
def find_edges(signal, filter=True, mode='bayes'):
    if filter == True:
        signal = median_filter(signal, size=5, mode='nearest')
        
    state_means, state_stdevs = get_state_means(signal)
    threshold = compute_threshold(state_means, state_stdevs, mode=mode)
    binarized = signal > threshold



## Can have LJ and NPX or LJ and camera or camera and NPX
## LJ, NPX --> NPX reference
## LJ, camera --> LJ reference
## camera, NPX --> NPX reference --> how do we find camera start?
def syncronize_timestamps_to_reference():
    """
    
    """

def timestamp_from_sample(reference_timestamps, samples):
    return reference_timestamps[samples]

## TODO -- name better
def identify_dropped_frames():
    pass

## TODO -- name better
def identify_dropped_flips():
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

## TODO -- find start and end first? there may be failure modes before the start and end of the experiment
## TODO -- move to signal processing submodule
## TODO -- add type hints for ndarrays?
def parse_frame_signal(frame_signal):
    """
    Processes photodoide signal. Assumes rising and falling edges represent frame transitions. 
    Signal is median filtered to remove momentary dropouts (<5 samples--not tested) while 
    preserving edges. Filtered signal is then binarized by `signal > mean` to facilitate
    edge detection (essentially finds midpoint of rising and falling edges). Note that this 
    will fail if the oscillations around LOW and HIGH are greater than half the separation
    of LOW and HIGH states. The median filtering helps significantly, but a sufficiently
    poor signal will result in noisy edge detection. 
    """
    filtered_signal = median_filter(frame_signal, size=5, mode='nearest')
    binarized = filtered_signal > filtered_signal.mean()
    edge_indices = np.where(np.diff(binarized)==1)[0]
    
    return edge_indices