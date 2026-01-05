import logging 

import numpy as np
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

@module(name='identifyCandidateSaccadesByPeak')
@produces()
@param("velocity_threshold_percentile", description="Velocity percentile to be considered a candidate saccade in pixels/s. Default is 95th percentile", default=95)
@param("distance_threshold_sampels", description="Minimum number of samples between peaks. Can be calculated using <time in seconds> * <camera_fps>. Default is 0.07s * 150fps = 10.5 samples", default=10.5)
def run(pupil_nt, velocity_threshold_percentile, distance_threshold_samples):
    # 1. calc horizontal velocity
    # 2. peak detection --> this is literally the entire extraction process
    ## TODO -- need to understand and significantly revise the code to remove all the interps (unless they're really needed). also remove appends because that's gonna be real slow. NOTE: I don't think we get any benefit from interpolating here since the waveform should look identical, even if we have "higher" resolution. Interpolation *may* have additional benefit for saccade timing, since there will be many ephys samples between each camera frame. we lose any benefit anyway cause we align to nearest frame in next step 
    ## TODO -- probably want to do timestamp alignment first, so we can just directly compute saccade timestamps
    ## TODO -- possibly roll in with classification because detection is dead simple (probably not, since this couples detection with classification)
    horizontal_velocity = np.diff(pupil_nt)
    velocity_threshold = np.nanpercentile(horizontal_velocity, velocity_threshold_percentile) ## TODO -- doubt this works real well for abnormal saccade distributions
    peak_indices, peak_properties = find_peaks(np.abs(horizontal_velocity), height=velocity_threshold, distance=distance_threshold_samples)

    return peak_indices