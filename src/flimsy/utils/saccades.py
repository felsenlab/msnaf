import logging

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter


logger = logging.getLogger(__name__)

def mask_low_confidence_samples(pose, threshold=1.0):
    confidence = pose[:,2]
    confidence_mask = confidence < threshold
    pose[confidence_mask, 0] = np.nan
    pose[confidence_mask, 1] = np.nan
    return pose

def identify_dropped_frames(interframe_intervals, nframes, framerate, threshold):
    if len(interframe_intervals) != nframes:
        logger.warning(f'Different number of frames ({nframes}) and timestamps ({len(interframe_intervals)})')

    expected_interval = 1e9 / framerate ## 1e9 = nanoseconds
    ratio = interframe_intervals/expected_interval
    dropped_indices = np.where(ratio > threshold)[0]

    to_insert = []
    for index in dropped_indices:
        n_dropped_frames = int(np.round(ratio[index], 0) - 1)

        logger.debug(index)
        logger.debug(index+n_dropped_frames)
        logger.debug(ratio[index])

        to_insert.extend(list(range(index, index+n_dropped_frames)))

    logger.warning(f'dropped_indices: {dropped_indices}')
    logger.debug(f'to_insert: {to_insert}')
    return dropped_indices, to_insert

def interpolate_gaps(pose):
    n_samples, n_columns = pose.shape
    interpolated = np.copy(pose)
    for column_index in range(n_columns):
        values = pose[:,column_index]
        indices_to_interp = np.where(np.isnan(values))[0]
        indices = np.arange(0, n_samples)
        ## NOTE: x is target indices to interp, xp and fp are real data to interp from
        logger.debug(indices_to_interp)
        logger.debug(indices.shape)
        logger.debug(values.shape)
        
        logger.info(f"to_interp: {indices_to_interp}")
        if len(indices_to_interp) > 0:
            interpolated_values = np.interp(x=indices_to_interp, xp=indices[~indices_to_interp], fp=values[~indices_to_interp]) 
            logger.debug(f"interp_values: {interpolated_values}")
            interpolated[indices_to_interp, column_index] = interpolated_values
    logger.warning(np.isnan(interpolated).sum())
    return interpolated

# TODO -- intervals have sub_ms jitter (+-0.001 ms), consistently 0.0003 ms higher than theoretical perfect
def insert_frames(projections, to_insert):
    logger.debug(projections.shape)
    logger.debug(type(projections))
    logger.debug(to_insert)

    value = np.nan
    corrected = np.insert(projections, to_insert, value, axis=0)

    return corrected

def smooth_signal(signal, window_size, method='gaussian'):
    if method == 'gaussian':
        #logger.debug(f"sigma: {sigma}, window_size: {window_size}")
        smooth = gaussian_filter1d(signal, sigma=window_size, axis=0)
    elif method == 'median':
        smooth = median_filter(signal, size=window_size)
        
    return smooth
