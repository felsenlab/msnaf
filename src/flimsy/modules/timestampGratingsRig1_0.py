import logging

import pandas as pd
import numpy as np

from flimsy.pipeline.basemodule import *
from flimsy.utils.ioer import find_files_matching_pattern, parse_experiment_file
from flimsy.utils.signal_processing import filter_dropout, match

logger = logging.getLogger(__name__)

@module(name="timestampGratingsRig1_0") 
@requires("metadata/basepath")
@requires("labjack/stimulus/raw")
@param("file_pattern")
@param("sample_rate")
@param("fallback")
@param("tolerance")

@produces("stimuli/dg/motion/timestamps")
@produces("stimuli/dg/grating/motion")
@produces("stimuli/dg/grating/timestamps")
@produces("stimuli/dg/probe/timestamps")
@produces("stimuli/dg/probe/motion")
@produces("stimuli/dg/probe/contrast")
@produces("stimuli/dg/probe/phase")
@produces("stimuli/dg/iti/timestamps")
@produces("stimuli/dg/cumtime")

def run(data, params):
    basepath = data["metadata/basepath"]
    photologic_signal = data["labjack/stimulus/raw"]

    file_pattern = params["file_pattern"]
    sample_rate = params["sample_rate"]
    fallback = params["fallback"]
    tolerance = params["tolerance"]

    metadata_files = find_files_matching_pattern(basepath, file_pattern, recursive=True)
    if len(metadata_files) != 1:
        logger.error(f"Got unexpected number of metadata files ({len(metadata_files)}) matching {file_pattern} in {basepath}. {metadata_files}")

    #stimulus_metadata = pd.read_csv(metadata_files.pop(), skiprows=4)
    meta, stimulus_metadata = parse_experiment_file(metadata_files.pop())
    logger.debug(meta)
    logger.debug(stimulus_metadata.shape)
    logger.debug(stimulus_metadata['Timestamp'])
    
    reference = stimulus_metadata['Timestamp'].to_numpy(dtype=np.float64)
    reference = reference - reference[0]
    filtered, mask = filter_dropout(photologic_signal, )
    inds_rise = np.where(np.diff(filtered.astype(int)) == 1)[0] + 1
    inds_fall = np.where(np.diff(filtered.astype(int)) == -1)[0] + 1

    events = inds_rise
    logger.debug(f'events: {events.shape}')

    pairs, times, region_quality, metrics = match(reference, events, sample_rate=sample_rate, tolerance=tolerance, fallback=fallback)


    #event_time_in_lj = frametimestamps[event_index_in_camera_frames]
    nanmask = np.isnan(times)
    if nanmask.sum() > 0:
        logger.warning("Some times are nan ({nanmask})")
    stimulus_metadata.loc[nanmask] = np.nan
    event_ids = stimulus_metadata["Event (1=Grating, 2=Motion, 3=Probe, 4=ITI)"].to_numpy()
    event_orientation = stimulus_metadata["Motion direction"].to_numpy()
    probe_contrast = stimulus_metadata["Probe contrast"].to_numpy()
    probe_phase = stimulus_metadata["Probe phase"].to_numpy()

    res = {}
    res["stimuli/dg/grating/motion"] = event_orientation[np.where(event_ids==2)]
    res["stimuli/dg/grating/timestamps"] = times[np.where(event_ids==1)]
    res["stimuli/dg/motion/timestamps"] = times[np.where(event_ids==2)]
    res["stimuli/dg/probe/timestamps"] = times[np.where(event_ids==3)]
    res["stimuli/dg/probe/motion"] = event_orientation[np.where(event_ids==3)]
    res["stimuli/dg/probe/contrast"] = probe_contrast[np.where(event_ids==3)]
    res["stimuli/dg/probe/phase"] = probe_phase[np.where(event_ids==3)]
    res["stimuli/dg/iti/timestamps"] = times[np.where(event_ids==4)]
    res["stimuli/dg/cumtime"] = np.cumsum(stimulus_metadata["Timestamp"].to_numpy())
    res["stimuli/dg/metrics"] = metrics

    return res

    ## Event (1=Grating, 2=Motion, 3=Probe, 4=ITI)
    # 1 = Appears
    # 2 = Starts moving
    # 3 = Probe shown
    # 4 = End block, start inter-trial-interval



# stimuli/dg/grating/motion
# stimuli/dg/grating/timestamps
# stimuli/dg/motion/timestamps
# stimuli/dg/iti/timestamps
# stimuli/dg/probe/motion
# stimuli/dg/probe/timestamps