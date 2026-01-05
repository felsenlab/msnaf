import logging

from flimsy.pipeline.registry import module
from flimsy.utils.ioer import load_pickle, load_from_h5, save_to_h5

logger = logging.getLogger(__name__)

@module(name='classifyCandidateSaccades')
def run(saccade_classifier_path, saccade_duration_regressor):
    ## load candidate waveforms
    ## load models
    ## normalize to peak velocity
    ## predict candidate identity --> clf.predict(norm_velocity)
    ## estimate saccade onset and offset --> reg.predict(norm_velocity)
    ## save out

    candidate_waveforms = load_from_h5(field)
    saccade_type_classifier = load_pickle(cls_path)
    onset_offset_regressor = load_pickle(reg_path)

    velocity_waveform = np.diff(candidate_waveforms) ## TODO -- is this horizontal only?
    normalized_waveforms = normalize_waveforms_by_velocity(velocity_waveform)

    predicted_labels = saccade_type_classifier.predict(normalized_waveforms)
    predicted_onsets, predicted_offsets = onset_offset_regressor.predict(normalized_waveforms)
    ## TOOD -- save out

    for field in {}:
        save_to_h5(field)

def normalize_waveforms_by_velocity(waveform):
    return waveform / np.abs(waveform).max(axis=1).reshape(-1, 1)


def extract_waveforms():
    #candidate_waveforms = np.zeros((len(peak_indices), window_size))
    waveform_indices = inds[:, None] + np.arange(-window_size, window_size)
    waveform_timestamps = timestamps[waveform_indices]
    candidate_waveform_x = horizontal_eye_position[waveform_indices]
    candidate_waveform_y = vertical_eye_position[waveform_indices]
    ## TODO -- combine waveforms