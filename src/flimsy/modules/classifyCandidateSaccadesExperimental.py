import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from flimsy.pipeline.basemodule import *
from flimsy.utils.ioer import load_pickle
from flimsy.utils.saccades import compute_template_match_scores

logger = logging.getLogger(__name__)

@module(name='classifyCandidateSaccadesExperimental', description="NOTE: Saccade onsets and offsets are aligned to the nearest frame, thus limiting the maximum temporal resolution to 1/fps, or ~7 ms at 150fps.\nNOTE: Classifiers only consider normalize horizontal velocity. They are not sensitive to amplitude, vertical velocity, or any other features")
# @requires("saccades/putative/left/indices")
# @requires("saccades/putative/right/indices")
# @requires("pose/rotated/left/pupil")
# @requires("pose/rotated/right/pupil")

@requires("saccades/putative/{side}/indices")
@requires("pose/smoothed/{side}")
        
@param("fieldnames")

            
# @requires("pose/smoothed/left")
# @requires("pose/smoothed/right")

@requires("labjack/cameras/timestamps")
#@produces("saccades/predicted/{side}/labels", description="Classification for each candidate waveform. Labels are 1, 0, or 1 for n/t, noise, or n/t respectively.")
#@produces("saccades/predicted/{side}/epochs", description="Frame index of saccade onsets for each candidate waveform") ## TODO -- maybe exclude noise events?
#@produces("saccade_offset", description="Frame index of saccade offsets for each candidate waveform") ## TODO -- maybe exclude noise events?
@param("window_size_time", default=0.2, description="Time to extract on each side of a candidate event. Default is 0.2, units unknown")
@param("n_samples", default=51, description="Number of points to use for resampling candidate waveforms")
@param("saccade_duration_regressor", description="")
#@param("saccade_classifier_path", description="")
@param("template_path")

@produces("saccades/template_matching/{side}/labels")
@produces("saccades/template_matching/{side}/epochs")

# @produces("saccades/template_matching/left/labels")
# @produces("saccades/template_matching/right/labels")

# @produces("saccades/template_matching/left/epochs")
# @produces("saccades/template_matching/right/epochs")

def run(data, params):
    window_size_time = params["window_size_time"]
    n_samples = params["n_samples"]
    frametimestamps = data["labjack/cameras/timestamps"]
    template = np.load(params["template"])

    #saccade_classifier_path = params["saccade_classifier_path"]
    saccade_duration_regressor = params["saccade_duration_regressor"]

    #saccade_type_classifier = load_pickle(saccade_classifier_path)
    onset_offset_regressor = load_pickle(saccade_duration_regressor)

    res = {}
    for side in params["fieldnames"]["side"]:
        #pupil_nt_pose = data[f"pose/rotated/{side}/pupil"]
        pupil_nt_pose = data[f"pose/smoothed/{side}"]
        candidate_saccade_indices = data[f"saccades/putative/{side}/indices"]
        
        ## 1. Extract candidate waveforms
        ## 2. load models -- one to classify waveforms as saccades/noise and one to predict when saccade starts and stops (onset and offset)
        ## 3. normalize waveforms by peak velocity
        ## 4. predict candidate identity --> clf.predict(norm_velocity)
        ## 5. estimate saccade onset and offset --> reg.predict(norm_velocity)

        logger.debug(pupil_nt_pose.shape)

        #candidate_waveforms = extract_waveforms(pupil_nt_pose, candidate_saccade_indices, window_size_samples)
        candidate_waveforms, timepoints = extract_and_resample_waveforms(pupil_nt_pose, candidate_saccade_indices, frametimestamps, window_size_time, n_samples)

        logger.error(timepoints.shape)

        logger.debug(candidate_waveforms.shape)
        logger.debug(candidate_saccade_indices.shape)

        position_template_scores = compute_template_match_scores(candidate_waveforms[:,:,0], template)
        velocity_template_scores = compute_template_match_scores(np.diff(candidate_waveforms[:,:,0], axis=1), np.diff(template))
        features = extract_features(candidate_waveforms[:,:,0])
        features_df = pd.DataFrame(features, index=range(len(candidate_waveforms)), columns=['sign', 'sym', 'ptp', 'se', 'gc'])
        dx_features, template_norm = derivative_energy_features(candidate_waveforms[:,:,0], template, sg_window=5, sg_order=3, energy_mode="abs")

        # score_df = pd.DataFrame(index=range(len(candidate_saccade_indices)), columns=['pos_ncc', 'vel_ncc'])
        # score_df['pos_ncc'] = position_template_scores['ncc']
        # score_df['vel_ncc'] = velocity_template_scores['ncc']
        score_df = pd.DataFrame(index=range(len(candidate_saccade_indices)))
        score_df['ncc'] = position_template_scores['ncc']
        score_df['dx_ncc'] = dx_features['energy_ncc'] #velocity_template_scores['ncc'] 
        score_df['wncc'] = velocity_template_scores['ncc'] * np.log(np.abs(position_template_scores['ptp']))
        #score_df['gc'] = features_df['gc']

        logger.error(score_df.isna().sum())

        nt_inds = np.where((position_template_scores['ncc'].squeeze()<=-0.75) & (velocity_template_scores['ncc'].squeeze()<-0.5) & (position_template_scores['ptp'].squeeze()>3.75))[0]
        tn_inds = np.where((position_template_scores['ncc'].squeeze()>=0.75) & (velocity_template_scores['ncc'].squeeze()>0.5) & (position_template_scores['ptp'].squeeze()>3.75))[0]
        #noise_inds = np.where((np.abs(score_df['ncc'].squeeze())<.6) | (np.abs(score_df['ncc'].squeeze())<0.5) | (score_df['ptp'].squeeze()<3))[0]

        y = np.zeros(len(score_df))
        y[nt_inds] = -1
        y[tn_inds] = 1

        lr = LogisticRegression(C=0.9)
        lr.fit(score_df, y)
        predicted_labels = lr.predict(score_df)

        #predicted_labels = saccade_type_classifier.predict(score_df)

        logger.debug(f"nlabels: {len(predicted_labels)}")
        logger.debug(f"nwaveforms: {len(candidate_waveforms)}")
        logger.debug(f"nputativesaccades: {len(candidate_saccade_indices)}")

        real = np.where(predicted_labels != 0)[0]

        logger.debug(f"realsaccadeindices: {real[-10:]}")

        horizontal_velocity = np.diff(candidate_waveforms[real,:,0], axis=1) ## TODO -- is this horizontal only?
        row_max = np.max(np.abs(horizontal_velocity), axis=1, keepdims=True)
        normalized_velocity = horizontal_velocity / row_max
        predicted_epochs = onset_offset_regressor.predict(normalized_velocity)

        absolute_epochs = get_absolute_saccade_epochs(candidate_saccade_indices[real], predicted_epochs, timepoints[real])
        absolute_indices = get_frames_from_timestamps(absolute_epochs, frametimestamps)

        logger.debug(f"indices: {absolute_indices[-20:]}")

        res[f"saccades/template_matching/{side}/labels"] = predicted_labels[real]
        res[f"saccades/template_matching/{side}/debuglabels"] = predicted_labels
        res[f"saccades/template_matching/{side}/timestamps"] = absolute_epochs
        res[f"saccades/template_matching/{side}/indices"] = absolute_indices
        res[f"saccades/template_matching/{side}/epochs"] = predicted_epochs
        res[f"saccades/template_matching/{side}/waveforms"] = candidate_waveforms[real]
        res[f"saccades/template_matching/{side}/ncc"] = position_template_scores['ncc']
        res[f"saccades/template_matching/{side}/dx_ncc"] = velocity_template_scores['ncc']
        res[f"saccades/template_matching/{side}/dx_energy"] = dx_features['energy_ncc']
        res[f"saccades/template_matching/{side}/ptp"] = position_template_scores['ptp']
        res[f"saccades/template_matching/{side}/ptpr"] = features_df['ptp'].to_numpy()
        res[f"saccades/template_matching/{side}/gc"] = features_df['gc'].to_numpy()
        res[f"saccades/template_matching/{side}/sym"] = features_df['sym'].to_numpy()
        res[f"saccades/template_matching/{side}/sign"] = features_df['sign'].to_numpy()
        res[f"saccades/template_matching/{side}/se"] = features_df['se'].to_numpy()

    return res


def get_frames_from_timestamps(indices, timestamps):
    frameindices = np.full(indices.shape, np.nan)
    frameindices[:,0] = np.interp(indices[:,0], timestamps, range(timestamps.size))
    frameindices[:,1] = np.interp(indices[:,1], timestamps, range(timestamps.size))

    return frameindices

def normalize_waveforms_by_velocity(waveforms):
    return waveforms / np.abs(waveforms).max(axis=1).reshape(-1, 1)

def extract_waveforms(signal, event_indices, window_size):
    waveform_indices = event_indices[:, None] + np.arange(-window_size, window_size)

    logger.debug(waveform_indices)

    waveforms = signal[waveform_indices]

    return waveforms

def get_absolute_saccade_epochs(saccade_indices, epochs, resampled_timestamps):
    epoch_timestamps = np.full((epochs.shape), np.nan)
    for i, peak_index in enumerate(saccade_indices):
        t = resampled_timestamps[i]
        midpoint = (t.size-1)/2
        midtime = np.interp(midpoint, np.arange(t.size), t)
        epoch_timestamp = epochs[i] + midtime
        epoch_timestamps[i] = epoch_timestamp
        # logger.debug(f"i: {i}, peakindex: {peak_index}")
        # logger.debug(f"resampled_timepoints: {t}")
        # logger.debug(f"midpoint: {midpoint}")
        #logger.debug(f"midtime: {midtime}")
        # logger.debug(f"epoch_timestamp: {epoch_timestamp}")

    return epoch_timestamps

def extract_and_resample_waveforms(pupil_pos, candidate_saccade_indices, frametimestamps, window_size_time, n_samples):
    horizontal_eye_position = pupil_pos[:,0]
    vertical_eye_position = pupil_pos[:,1]

    resampled_waveforms = np.full((candidate_saccade_indices.size, n_samples + 1, 2), np.nan)
    resampled_timepoints = np.full((candidate_saccade_indices.size, n_samples + 1), np.nan)
    for i, peak_index in enumerate(candidate_saccade_indices):
        peak_timestamp = np.mean([frametimestamps[peak_index], frametimestamps[peak_index+1]])
        
        ## extract time window peak
        left_timestamp = peak_timestamp - window_size_time
        right_timestamp = peak_timestamp + window_size_time

        ## uniformly sample time in the window
        sample_times = np.linspace(left_timestamp, right_timestamp, n_samples + 1)

        ## sample times likely don't fall on frames, so interpolate the waveform at those times
        resampled_waveform_horizontal = np.interp(sample_times, frametimestamps[:horizontal_eye_position.size], horizontal_eye_position)
        resampled_waveform_vertical = np.interp(sample_times, frametimestamps[:vertical_eye_position.size], vertical_eye_position)
        resampled_waveforms[i,:,0] = resampled_waveform_horizontal
        resampled_waveforms[i,:,1] = resampled_waveform_vertical

        resampled_timepoints[i] = sample_times
        
    return resampled_waveforms, resampled_timepoints


import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import entropy

def robust_zscore(x):
    med = np.median(x)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    return (x - med) / (iqr + 1e-8)

def conditional_spectral_entropy(x, remove_frac=0.05):
    X = np.abs(np.fft.rfft(x))**2
    X /= X.sum()

    peak = np.argmax(X)
    bw = int(remove_frac * len(X))
    mask = np.ones(len(X), dtype=bool)
    mask[max(0, peak-bw):min(len(X), peak+bw)] = False

    X2 = X[mask]
    X2 /= X2.sum() + 1e-8

    return entropy(X2)

def peak_to_plateau(x):
    analytic = hilbert(x)
    envelope = np.abs(analytic)

    high = np.percentile(envelope, 90)
    low = np.percentile(envelope, 10)

    dx = np.abs(np.diff(x))
    plateau = np.percentile(dx, 20)

    return (high - low) / (plateau + 1e-8)

def symmetry_slope(x, scales=(1, 3, 7, 15)):
    sym = []
    for s in scales:
        if s > 1:
            kernel = np.ones(s) / s
            xs = np.convolve(x, kernel, mode="same")
        else:
            xs = x
        r = np.corrcoef(xs, xs[::-1])[0, 1]
        sym.append(r)
    scales = np.array(scales)
    sym = np.array(sym)
    return np.polyfit(scales, sym, 1)[0]

def derivative_sign_changes(x, cutoff_frac=0.1):
    """
    cutoff_frac: fraction of Nyquist for low-pass
    """
    n = len(x)
    b, a = butter(3, cutoff_frac)
    x_f = filtfilt(b, a, x)
    dx = np.diff(x_f)
    return np.sum(np.diff(np.sign(dx)) != 0)

def gradient_center_distance(x):
    dx = np.abs(np.diff(x))
    i = np.argmax(dx)
    center = (len(dx) - 1) / 2
    return abs(i - center) / center  # normalized [0, ~1]

def extract_features(X):
    feats = []
    for x in X:
        x = robust_zscore(x)
        feats.append([
            derivative_sign_changes(x),
            symmetry_slope(x),
            peak_to_plateau(x),
            conditional_spectral_entropy(x),
            gradient_center_distance(x)
        ])
    return np.array(feats)

import numpy as np
from scipy.signal import savgol_filter

def derivative_energy_features(
    events,
    template,
    *,
    dt=1.0,
    sg_window=11,
    sg_order=3,
    energy_mode="abs",   # "abs" or "square"
    eps=1e-12
):
    """
    Compute derivative-energy-based matching features for a batch of events.

    Parameters
    ----------
    events : ndarray, shape (n_events, n_samples)
        Windowed event signals.
    template : ndarray, shape (n_samples,)
        Template signal (same length as events).
    dt : float
        Sample spacing.
    sg_window : int
        Savitzky–Golay window length (must be odd).
    sg_order : int
        Savitzky–Golay polynomial order.
    energy_mode : {"abs", "square"}
        How to convert derivative to energy.
    eps : float
        Numerical stability constant.

    Returns
    -------
    features : dict
        Dictionary containing:
        - "energy_ncc": NCC between normalized derivative energy and template
        - "centroid": energy centroid (0–1 normalized time)
        - "spread": energy standard deviation (0–1)
        - "early_late_ratio": early vs late energy ratio
    """

    events = np.asarray(events)
    template = np.asarray(template)

    n_events, n_samples = events.shape

    # --- Derivative via Savitzky–Golay ---
    d_events = savgol_filter(
        events,
        window_length=sg_window,
        polyorder=sg_order,
        deriv=1,
        delta=dt,
        axis=1,
        mode="interp"
    )

    d_template = savgol_filter(
        template,
        window_length=sg_window,
        polyorder=sg_order,
        deriv=1,
        delta=dt,
        mode="interp"
    )

    # --- Energy construction ---
    if energy_mode == "abs":
        e_events = np.abs(d_events)
        e_template = np.abs(d_template)
    elif energy_mode == "square":
        e_events = d_events**2
        e_template = d_template**2
    else:
        raise ValueError("energy_mode must be 'abs' or 'square'")

    # --- Normalize energy (scale invariance) ---
    e_events_sum = np.sum(e_events, axis=1, keepdims=True) + eps
    e_events_norm = e_events / e_events_sum

    e_template_norm = e_template / (np.sum(e_template) + eps)

    # --- Energy NCC ---
    # zero-mean
    e_events_zm = e_events_norm - np.mean(e_events_norm, axis=1, keepdims=True)
    e_template_zm = e_template_norm - np.mean(e_template_norm)

    num = np.sum(e_events_zm * e_template_zm, axis=1)
    den = (
        np.linalg.norm(e_events_zm, axis=1)
        * np.linalg.norm(e_template_zm)
        + eps
    )

    energy_ncc = num / den

    # --- Time axis (normalized 0–1) ---
    t = np.linspace(0.0, 1.0, n_samples)

    # --- Energy centroid ---
    centroid = np.sum(e_events_norm * t, axis=1)

    # --- Energy spread (std) ---
    spread = np.sqrt(
        np.sum(e_events_norm * (t - centroid[:, None])**2, axis=1)
    )

    # --- Early / late energy ratio ---
    mid = 0.5
    early_energy = np.sum(e_events_norm[:, t <= mid], axis=1)
    late_energy  = np.sum(e_events_norm[:, t >  mid], axis=1)

    early_late_ratio = early_energy / (late_energy + eps)

    return {
        "energy_ncc": energy_ncc,
        "centroid": centroid,
        "spread": spread,
        "early_late_ratio": early_late_ratio,
    }, e_template_norm
