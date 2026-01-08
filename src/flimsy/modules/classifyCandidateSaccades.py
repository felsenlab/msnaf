import logging

import numpy as np

from flimsy.pipeline.basemodule import *
from flimsy.utils.ioer import load_pickle

logger = logging.getLogger(__name__)

@module(name='classifyCandidateSaccades', description="NOTE: Saccade onsets and offsets are aligned to the nearest frame, thus limiting the maximum temporal resolution to 1/fps, or ~7 ms at 150fps.\nNOTE: Classifiers only consider normalize horizontal velocity. They are not sensitive to amplitude, vertical velocity, or any other features")
@requires("saccades/putative/left/indices")
@requires("saccades/putative/right/indices")
@requires("pose/interpolated/left")
@requires("pose/interpolated/right")
@requires("labjack/cameras/timestamps")
#@produces("saccades/predicted/{side}/labels", description="Classification for each candidate waveform. Labels are 1, 0, or 1 for n/t, noise, or n/t respectively.")
#@produces("saccades/predicted/{side}/epochs", description="Frame index of saccade onsets for each candidate waveform") ## TODO -- maybe exclude noise events?
#@produces("saccade_offset", description="Frame index of saccade offsets for each candidate waveform") ## TODO -- maybe exclude noise events?
@param("window_size_samples", description="Number of samples to extract on each side of a candidate event in samples. Can be calculated using <time in seconds> * <camera_fps>. The default is 30, for 71 total samples", default=26)
@param("window_size_time", default=0.2, description="Time to extract on each side of a candidate event. Default is 0.2, units unknown")
@param("n_samples", default=51, description="Number of points to use for resampling candidate waveforms")
@param("saccade_classifier_path", description="")
@param("saccade_duration_regressor", description="")

@produces("saccades/predicted/left/labels")
@produces("saccades/predicted/right/labels")

@produces("saccades/predicted/left/epochs")
@produces("saccades/predicted/right/epochs")

def run(data, params):
    window_size_samples = params["window_size_samples"]
    window_size_time = params["window_size_time"]
    n_samples = params["n_samples"]
    saccade_classifier_path = params["saccade_classifier_path"]
    saccade_duration_regressor = params["saccade_duration_regressor"]
    frametimestamps = data["labjack/cameras/timestamps"]

    res = {}
    for side in ["left", "right"]:
        pupil_nt_pose = data[f"pose/interpolated/{side}"]
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

        saccade_type_classifier = load_pickle(saccade_classifier_path)
        onset_offset_regressor = load_pickle(saccade_duration_regressor)

        horizontal_velocity = np.diff(candidate_waveforms[:,:,0], axis=1) ## TODO -- is this horizontal only?
        #normalized_velocity = horizontal_velocity / np.abs(horizontal_velocity).max(axis=1).reshape(-1, 1)
        row_max = np.max(np.abs(horizontal_velocity), axis=1, keepdims=True)
        normalized_velocity = horizontal_velocity / row_max

        logger.debug(normalized_velocity.max())

        predicted_labels = saccade_type_classifier.predict(normalized_velocity)
        predicted_epochs = onset_offset_regressor.predict(normalized_velocity)

        # predicted_epochs_frames = np.full(predicted_epochs_time.shape, np.nan)
        # predicted_epochs_frames[:,0] = np.interp(predicted_epochs_time[:,0], )

        real = np.where(predicted_labels != 0)[0]

        absolute_epochs = get_absolute_saccade_epochs(candidate_saccade_indices[real], predicted_epochs[real], timepoints[real])
        absolute_indices = get_frames_from_timestamps(absolute_epochs, frametimestamps)

        logger.debug(f"Real saccade indices: {real} (n={real.shape[0]})")
        logger.debug(predicted_labels)

        res[f"saccades/predicted/{side}/labels"] = predicted_labels[real]
        res[f"saccades/predicted/{side}/timestamps"] = absolute_epochs
        res[f"saccades/predicted/{side}/indices"] = absolute_indices
        res[f"saccades/predicted/{side}/epochs"] = predicted_epochs[real]
        res[f"saccades/predicted/{side}/waveforms"] = candidate_waveforms[real]

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
        logger.debug(f"midtime: {midtime}")
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


# def run(pupil_nt_pose, candidate_saccade_indices, saccade_classifier_path, saccade_duration_regressor, window_size_samples):
#     ## 1. Extract candidate waveforms
#     ## 2. load models -- one to classify waveforms as saccades/noise and one to predict when saccade starts and stops (onset and offset)
#     ## 3. normalize waveforms by peak velocity
#     ## 4. predict candidate identity --> clf.predict(norm_velocity)
#     ## 5. estimate saccade onset and offset --> reg.predict(norm_velocity)

#     candidate_waveforms = extract_waveforms(pupil_nt_pose, candidate_saccade_indices, window_size_samples)

#     saccade_type_classifier = load_pickle(saccade_classifier_path)
#     onset_offset_regressor = load_pickle(saccade_duration_regressor)

#     horizontal_velocity = np.diff(candidate_waveforms) ## TODO -- is this horizontal only?
#     normalized_velocity = horizontal_velocity / np.abs(horizontal_velocity).max(axis=1).reshape(-1, 1)

#     predicted_labels = saccade_type_classifier.predict(normalized_velocity)
#     predicted_epochs = onset_offset_regressor.predict(normalized_velocity)

#     return predicted_labels, predicted_epochs