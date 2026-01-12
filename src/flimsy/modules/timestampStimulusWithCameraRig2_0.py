import logging

import pandas as pd
import numpy as np

from flimsy.pipeline.basemodule import *
from flimsy.utils.ioer import find_files_matching_pattern

logger = logging.getLogger(__name__)

@module(name="timestampStimulusWithCameraRig2_0") 
@requires("metadata/basepath")
@requires("labjack/cameras/timestamps")
@param("file_pattern")

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
    frametimestamps = data["labjack/cameras/timestamps"]

    file_pattern = params["file_pattern"]

    metadata_files = find_files_matching_pattern(basepath, file_pattern, recursive=True)
    if len(metadata_files) != 1:
        logger.error(f"Got unexpected number of metadata files ({len(metadata_files)}) matching {file_pattern} in {basepath}. {metadata_files}")

    stimulus_metadata = pd.read_csv(metadata_files.pop())
    all_nans_rows = stimulus_metadata.isna().all(axis=1)
    logger.debug(all_nans_rows[2210:2220])
    first_bad_row = np.argmax(all_nans_rows)
    logger.debug(first_bad_row)
    stimulus_metadata = stimulus_metadata.iloc[:first_bad_row]

    #logger.debug(stimulus_metadata.iloc[3000:3010])

    event_index_in_camera_frames = stimulus_metadata["Frame Number"].to_numpy(dtype=int)
    # for frame in dropped_frames:
    #     event_index_in_camera_frames[event_index_in_camera_frames>frame] += 1
    
    logger.debug(event_index_in_camera_frames[:20])
    logger.debug(type(event_index_in_camera_frames))
    logger.debug(frametimestamps[:20])

    event_time_in_lj = frametimestamps[event_index_in_camera_frames]
    event_ids = stimulus_metadata["Event (1=Grating, 2=Motion, 3=Probe, 4=ITI)"].to_numpy()
    event_orientation = stimulus_metadata["Motion direction"].to_numpy()
    probe_contrast = stimulus_metadata["Probe contrast"].to_numpy()
    probe_phase = stimulus_metadata["Probe phase"].to_numpy()

    res = {}
    res["stimuli/dg/grating/motion"] = event_orientation[np.where(event_ids==2)]
    res["stimuli/dg/grating/timestamps"] = event_time_in_lj[np.where(event_ids==1)]
    res["stimuli/dg/motion/timestamps"] = event_time_in_lj[np.where(event_ids==2)]
    res["stimuli/dg/probe/timestamps"] = event_time_in_lj[np.where(event_ids==3)]
    res["stimuli/dg/probe/motion"] = event_orientation[np.where(event_ids==3)]
    res["stimuli/dg/probe/contrast"] = probe_contrast[np.where(event_ids==3)]
    res["stimuli/dg/probe/phase"] = probe_phase[np.where(event_ids==3)]
    res["stimuli/dg/iti/timestamps"] = event_time_in_lj[np.where(event_ids==4)]
    res["stimuli/dg/cumtime"] = np.cumsum(stimulus_metadata["Timestamp"].to_numpy())

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