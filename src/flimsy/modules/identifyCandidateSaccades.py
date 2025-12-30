import logging 

logger = logging.getLogger(__name__)

@module(name='identifyCandidateSaccades')
def run():
    pass

    pose = load_from_h5(field)
    putative_saccade_waveforms, _, _ = detect_candidate_saccades(pose)
    ## load pose imformation
    ## calc horizontal velocity
    ## peak detection --> this is literally the entire extraction process
    ## TODO -- need to understand and significantly revise the code to remove all the interps (unless they're really needed). also remove appends because that's gonna be real slow. NOTE: I don't think we get any benefit from interpolating here since the waveform should look identical, even if we have "higher" resolution. Interpolation *may* have additional benefit for saccade timing, since there will be many ephys samples between each camera frame. we lose any benefit anyway cause we align to nearest frame in next step 

    ## TODO -- probably want to do timestamp alignment first, so we can just directly compute saccade timestamps

    ## TODO -- possibly roll in with classification because detection is dead simple (probably not, since this couples detection with classification)

def detect_candidate_saccades():
    height_threshold = np.nanpercentile(horizontal_eye_velocity, percentile) ## TODO -- doubt this works real well for abnormal saccade distributions
    distance_threshold = fps * minimum_peak_distance
    
    horizontal_eye_velocity = np.diff(horizontal_eye_position)
    peak_indices, properties = find_peaks(np.abs(horizontal_eye_velocity, height=height_threshold, distance=distance_threshold))

    #candidate_waveforms = np.zeros((len(peak_indices), window_size))
    waveform_indices = inds[:, None] + np.arange(-window_size, window_size)
    waveform_timestamps = timestamps[waveform_indices]
    candidate_waveform_x = horizontal_eye_position[waveform_indices]
    candidate_waveform_y = vertical_eye_position[waveform_indices]
    ## TODO -- combine waveforms