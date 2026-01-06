import logging

import numpy as np

from flimsy.pipeline.basemodule import *
from flimsy.utils.ioer import load_pickle, load_from_h5, save_to_h5

logger = logging.getLogger(__name__)

@module(name='classifyCandidateSaccades', description="NOTE: Saccade onsets and offsets are aligned to the nearest frame, thus limiting the maximum temporal resolution to 1/fps, or ~7 ms at 150fps.\nNOTE: Classifiers only consider normalize horizontal velocity. They are not sensitive to amplitude, vertical velocity, or any other features")
@requires()
@produces("saccades/predicted/{side}/labels", description="Classification for each candidate waveform. Labels are 1, 0, or 1 for n/t, noise, or n/t respectively.")
@produces("saccades/predicted/{side}/epochs", description="Frame index of saccade onsets for each candidate waveform") ## TODO -- maybe exclude noise events?
@produces("saccade_offset", description="Frame index of saccade offsets for each candidate waveform") ## TODO -- maybe exclude noise events?
@param("window_size_samples", description="Number of samples to extract on each side of a candidate event in samples. Can be calculated using <time in seconds> * <camera_fps>. The default is 30, for 71 total samples", default=30)
def run(pupil_nt_pose, candidate_saccade_indices, saccade_classifier_path, saccade_duration_regressor, window_size_samples):
    ## 1. Extract candidate waveforms
    ## 2. load models -- one to classify waveforms as saccades/noise and one to predict when saccade starts and stops (onset and offset)
    ## 3. normalize waveforms by peak velocity
    ## 4. predict candidate identity --> clf.predict(norm_velocity)
    ## 5. estimate saccade onset and offset --> reg.predict(norm_velocity)

    candidate_waveforms = extract_waveforms(pupil_nt_pose, candidate_saccade_indices, window_size_samples)

    saccade_type_classifier = load_pickle(saccade_classifier_path)
    onset_offset_regressor = load_pickle(saccade_duration_regressor)

    horizontal_velocity = np.diff(candidate_waveforms) ## TODO -- is this horizontal only?
    normalized_velocity = horizontal_velocity / np.abs(horizontal_velocity).max(axis=1).reshape(-1, 1)

    predicted_labels = saccade_type_classifier.predict(normalized_velocity)
    predicted_epochs = onset_offset_regressor.predict(normalized_velocity)

    return predicted_labels, predicted_epochs

def normalize_waveforms_by_velocity(waveforms):
    return waveforms / np.abs(waveforms).max(axis=1).reshape(-1, 1)

def extract_waveforms(signal, event_indices, window_size):
    waveform_indices = event_indices[:, None] + np.arange(-window_size, window_size)
    waveforms = signal[waveform_indices]

    return waveforms