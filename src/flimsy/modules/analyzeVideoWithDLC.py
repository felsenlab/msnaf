import logging

import deeplabcut as dlc

from flimsy.pipeline.basemodule import *
from flimsy.utils.ioer import find_files_matching_pattern

logger = logging.getLogger(__name__)


@module(name="extractPoseWithDLC", description="This module will find all video files matching the provided pattern in the session folder, regardless of folder structure (it will search all subfolders for the video pattern)")
@requires("base_path", description="Path to session folder") ## TODO -- this was probably already provided earlier, should we get this from the h5?
@param("video_pattern", description="String that will be used to identify video files. The wildcard operator * matches any number of characters (including none) in a file name, so a pattern like file_pattern='*Cam*.mp4' will select all files that contain 'Cam' and end with '.mp4', such as '20251112_unitME_session002_leftCam-0000.mp4' or '20251112_unitME_session002_rightCam-0000.mp4'; similarly, ? matches exactly one character.")
@param("dlc_path", description="Path to the deeplabcut config file for the model to be used for analysis. This is typically a config.yaml in the deeplabcut project folder.")
#@produces(group="pose/raw", type="dynamic") ## produces nothing
def run(base_path, video_pattern, dlc_config_path):
    video_list = find_files_matching_pattern(base_path, video_pattern)
    dlc.analyze_videos(dlc_config_path, video_list, save_as_csv=True)

    ## TODO -- delete extraneous files?
    
    

