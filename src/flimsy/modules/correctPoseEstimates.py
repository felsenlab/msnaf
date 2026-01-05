import logging

import numpy as np
from scipy.ndimage import gaussian_filter1d

from flimsy.pipeline.basemodule import *

logger = logging.getLogger(__name__)

@module(name='correctPoseEstimates')
@produces('test')
@param("smooth", description="", default=True)
@param("framerate", default=150)
@param("framedrop_threshold", default=1.5)
@param("confidence_threshold", default=1.0)
def run(pose, confidence_threshold, frame_intervals, framerate, framedrop_threshold, normalize=False, center=True, smooth=True):
    # 1. blank low confidence estiamtes --> maybe zscore not raw threshold?
    # 2. rotate / project onto nasal-temporal axis and center
    # 3. find and blank dropped frames
    # 4. interpolate gaps (low confidence + dropped frames)
    ## TODO -- previous note that nans were being introduced at high-confidence samples
    ## TODO -- do we want to median filter to remove high frequency noise?
    pupil_pose, nasal_pose, temporal_pose, dorsal_pose, ventral_pose = pose ## TODO -- this is not robust at all

    masked_pupil_pose = mask_low_confidence_samples(pupil_pose, threshold=confidence_threshold)
    masked_nasal_pose = mask_low_confidence_samples(pupil_pose, threshold=confidence_threshold)
    masked_temporal_pose = mask_low_confidence_samples(nasal_pose, threshold=confidence_threshold)
    masked_dorsal_pose = mask_low_confidence_samples(dorsal_pose, threshold=confidence_threshold)
    masked_ventral_pose = mask_low_confidence_samples(ventral_pose, threshold=confidence_threshold)

    corrected_pupil_pos = compute_pupil_projections(masked_pupil_pose, masked_nasal_pose, masked_temporal_pose, masked_dorsal_pose, masked_ventral_pose, normalize=normalize, center=center)
    frames_to_insert = identify_dropped_frames(frame_intervals, len(corrected_pupil_pos), framerate, framedrop_threshold)
    indexcorrected_pose = insert_frames(corrected_pupil_pos, frames_to_insert)
    imputed_pose = interpolate_gaps(indexcorrected_pose)
    if smooth:
        pose_estimates = smooth_signal(imputed_pose)
    else: 
        pose_estimates = imputed_pose

    return pose_estimates

def mask_low_confidence_samples(pose, confidence, threshold=1.0):
    confidence_mask = confidence < threshold
    pose[confidence_mask, [0,1]] = np.nan
    return pose

# def rotate_and_center():
#     ## TODO -- understand the math better. maybe be able to avoid mean subtraction if we use single origin
#     mean_nasal_pos = np.nanmean(nasal_pose, axis=0)
#     mean_temporal_pos = np.nanmean(nasal_pose, axis=0)
#     mean_dorsal_pos = np.nanmean(nasal_pose, axis=0)
#     mean_ventral_pos = np.nanmean(nasal_pose, axis=0)

#     mean_subtracted =  
#     horizontal_projection = np.dot(, )

def compute_pupil_projections(
    pupil_xy,
    nasal_xy,
    temporal_xy,
    dorsal_xy,
    ventral_xy,
    normalize=False,
    center=True):
    """
    Project pupil center onto nasal–temporal and dorsal–ventral axes using separate landmark arrays.

    Parameters
    ----------
    pupil_center : ndarray, shape (N, 2)
        Pupil center positions over time or frames.
    nasal_landmark : ndarray, shape (N, 2)
        Nasal landmark positions over time.
    temporal_landmark : ndarray, shape (N, 2)
        Temporal landmark positions over time.
    dorsal_landmark : ndarray, shape (N, 2)
        Dorsal (upper) landmark positions over time.
    ventral_landmark : ndarray, shape (N, 2)
        Ventral (lower) landmark positions over time.
    normalize : bool, default False
        If True, projections are divided by the corresponding eye axis length
        (nasal–temporal for horizontal, dorsal–ventral for vertical),
        yielding unitless values expressed as fractions of eye size.
    center : bool, default True
        If True, subtract the mean projection over time for each axis,
        so each series is zero-centered.

    Returns
    -------
    proj : ndarray, shape (N, 2)
        Column 0: nasal (−) to temporal (+) projection of the pupil center.
        Column 1: dorsal (−) to ventral (+) projection of the pupil center.

    Notes
    -----
    - The projections are computed as **signed scalar distances** along each axis.
    Positive values indicate motion toward temporal or ventral directions,
    negative values indicate motion toward nasal or dorsal directions.
    - Using separate arrays rather than a dict simplifies vectorized operations
    and avoids dictionary key lookups.
    """

    # Mean landmark positions (robust to NaNs)
    nasal   = np.nanmean(nasal_xy,   axis=0)
    temporal= np.nanmean(temporal_xy,axis=0)
    dorsal  = np.nanmean(dorsal_xy,  axis=0)
    ventral = np.nanmean(ventral_xy, axis=0)

    # Axis vectors
    nt = temporal - nasal           # nasal -> temporal
    dv = ventral - dorsal           # dorsal -> ventral

    nt_len = np.linalg.norm(nt)
    dv_len = np.linalg.norm(dv)

    if nt_len == 0 or dv_len == 0:
        raise ValueError("Degenerate eye axis (zero length)")

    # Unit vectors define the coordinate system
    nt_hat = nt / nt_len
    dv_hat = dv / dv_len

    # Use eye centroid as a common origin
    origin = 0.25 * (nasal + temporal + dorsal + ventral)

    # Vectors from origin to pupil
    w = pupil_xy - origin    # shape (N, 2)

    # Signed scalar projections
    horiz = np.dot(w, nt_hat)   # nasal (−) → temporal (+)
    vert  = np.dot(w, dv_hat)   # dorsal (−) → ventral (+)

    proj = np.column_stack([horiz, vert])

    if normalize:
        proj[:, 0] /= nt_len
        proj[:, 1] /= dv_len

    if center:
        proj -= np.nanmean(proj, axis=0)

    return proj ## TODO -- return nasal-temporal and dorsal-ventral projections separately

def identify_dropped_frames(interframe_intervals, nframes, framerate, threshold):
    if len(interframe_intervals) != nframes:
        raise Exception('Different number of frames ({nframes}) and timestamps ({len(interframe_intervals)})')

    expected_interval = 1e9 / framerate ## 1e9 = nanoseconds
    ratio = interframe_intervals/expected_interval
    dropped_indices = np.where(ratio > threshold)[0]

    to_insert = []
    for index in dropped_indices:
        n_dropped_frames = np.round(ratio[index], 0) - 1
        to_insert.append(range(index, index+n_dropped_frames))

    return dropped_indices, to_insert

# TODO -- intervals have sub_ms jitter (+-0.001 ms), consistently 0.0003 ms higher than theoretical perfect
def insert_frames(projections, to_insert):
    value = np.array([np.nan, np.nan])
    corrected = np.insert(projections, to_insert, value, axis=0)

    return corrected

def smooth_signal(signal, window_size):
    sigma = np.round(fps * window_size, 2)
    return gaussian_filter1d(signal, sigma=sigma, axis=0)

def interpolate_gaps(pose):
    n_samples, n_columns = pose.shape
    interpolated = np.copy(pose)
    for column_index in range(n_columns):
        values = pose[:,column_index]
        indices_to_interp = np.where(np.isnan(values))
        indices = np.arange(0, n_samples)
        ## NOTE: x is target indices to interp, xp and fp are real data to interp from
        interpolated_values = np.interp(x=indices_to_interp, xp=indices, fp=pose) 
        interpolated[indices_to_interp, column_index] = interpolated_values
    return interpolated