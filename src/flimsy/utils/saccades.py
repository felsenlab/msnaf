import logging

import numpy as np
import pandas as pd
from hampel import hampel
from scipy.signal import savgol_filter
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
        to_insert.extend(list(range(index, index+n_dropped_frames)))

    logger.warning(f'Found {dropped_indices.size} dropped frames at {dropped_indices}')

    return dropped_indices, to_insert

def interpolate_gaps(pose):
    n_samples, n_columns = pose.shape
    interpolated = np.copy(pose)
    for column_index in range(n_columns):
        values = pose[:,column_index]
        indices_to_interp = np.where(np.isnan(values))[0]
        indices = np.arange(0, n_samples)
        ## NOTE: x is target indices to interp, xp and fp are real data to interp from
        
        #logger.info(f"to_interp: {indices_to_interp}")
        if len(indices_to_interp) > 0:
            mask = np.ones(values.size, dtype=bool)
            mask[indices_to_interp] = 0
            
            interpolated_values = np.interp(x=indices_to_interp, xp=indices[mask], fp=values[mask]) 
            logger.warning(f"Found {indices_to_interp.size} masked indices to interpolate (column: {column_index})")
            logger.debug(f"nans in fp: {np.isnan(values[~indices_to_interp]).sum()}")
            interpolated[indices_to_interp, column_index] = interpolated_values
    logger.warning(np.isnan(interpolated).sum())
    return interpolated

# TODO -- intervals have sub_ms jitter (+-0.001 ms), consistently 0.0003 ms higher than theoretical perfect
def insert_frames(projections, to_insert):

    value = np.nan
    corrected = np.insert(projections, to_insert, value, axis=0)

    return corrected

def smooth_signal(signal, window_size, method='gaussian', n_sigma=3.0):
    if method == 'gaussian':
        #logger.debug(f"sigma: {sigma}, window_size: {window_size}")
        smooth = gaussian_filter1d(signal, sigma=window_size, axis=0)
    elif method == 'median':
        smooth = median_filter(signal, size=window_size)
    elif method == "hampelsavgol":
        smooth = np.full(signal.shape, np.nan)
        for i_col in range(signal.shape[1]):
            smooth[:,i_col] = hampel(signal[:,i_col], window_size=window_size, n_sigma=n_sigma).filtered_data
        #smooth = median_filter(signal, size=(3,1))
        smooth = savgol_filter(signal, window_length=11, polyorder=3, axis=0)
        
    return smooth

def compute_template_match_scores(
    events,
    template,
    *,
    eps=1e-8,
):
    """
    Center-restricted sigmoid scoring.

    Parameters
    ----------
    X : (n_events, window_size)
        Event-wise signals with centered inflection
    template : (k,)
        Centered sigmoid template

    Returns
    -------
    scores : dict of arrays, each shape (n_events,)
    """

    X = np.asarray(events)
    t = np.asarray(template)

    n, m = X.shape
    k = len(t)

    if k > m:
        raise ValueError("template longer than event window")

    # ---- Extract centered window ----
    start = (m - k) // 2
    stop = start + k
    W = X[:, start:stop]           # (n, k)

    # ---- Peak-to-peak amplitude ----
    ptp = np.ptp(W[:,:], axis=1) #was 20:30

    # ============================================================
    # NCC (zero-mean, scale-invariant)
    # ============================================================
    t0 = t - t.mean()
    t_norm = np.linalg.norm(t0) + eps

    W0 = W - W.mean(axis=1, keepdims=True)
    W_norm = np.linalg.norm(W0, axis=1) + eps

    ncc = np.einsum("ij,j->i", W0, t0) / (W_norm * t_norm)

    # ============================================================
    # Gradient-based structure
    # ============================================================
    dW = np.gradient(W, axis=1)
    dt = np.gradient(t)

    # ---- Gradient matching score ----
    dt0 = dt - dt.mean()
    dt_norm = np.linalg.norm(dt0) + eps

    dW0 = dW - dW.mean(axis=1, keepdims=True)
    dW_norm = np.linalg.norm(dW0, axis=1) + eps

    gradient_score = np.einsum("ij,j->i", dW0, dt0) / (dW_norm * dt_norm)

    # ---- Monotonicity score ----
    monotonicity_score = np.mean(dW > 0, axis=1)

    # ---- Symmetry score (derivative symmetry about center) ----
    mid = k // 2
    left = dW[:, :mid]
    right = dW[:, mid + 1:][:, ::-1]

    L = min(left.shape[1], right.shape[1])
    left = left[:, -L:]
    right = right[:, :L]

    num = np.sum((left - right) ** 2, axis=1)
    den = np.sum(left ** 2 + right ** 2, axis=1) + eps

    symmetry_score = 1.0 - num / den
    symmetry_score = np.clip(symmetry_score, 0.0, 1.0)

    return {
        "ncc": ncc,
        "ptp": ptp,
        "gradient_score": gradient_score,
        "monotonicity_score": monotonicity_score,
        "symmetry_score": symmetry_score,
    }