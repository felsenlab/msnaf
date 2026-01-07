import pandas as pd

from flimsy.pipeline.basemodule import *
from flimsy.utils.ioer import find_files_matching_pattern
from flimsy.utils.validation import check_scorer_matches_map

logger = logging.getLogger(__name__)

@module(name="loadPoseEstimates", description="This module will find all deeplabcut output files matching the provided pattern in the session folder, regardless of folder structure (it will search all subfolders for the file pattern)")
@requires("metadata/basepath", description="Path to session folder") ## TODO -- this was probably already provided earlier, should we get this from the h5?
@param("file_pattern", description="String that will be used to identify deeplabcut output files. The pipeline is designed to load the h5 output, rather than csvs. The wildcard operator * matches any number of characters (including none) in a file name, so a pattern like video_pattern='*DLC*.csv' will select all files that contain 'DLC' and end with '.csv', such as '20251217_unitME_session009_leftCam-0000DLC_resnet101_sacnetJan29shuffle1_1030000.csv' or '20251217_unitME_session009_rightCam-0000DLC_resnet101_sacnetJan29shuffle1_1030000.csv'; similarly, ? matches exactly one character.")
#@param("mapfile", description="Path to a configuration yaml file that defines the columns output by the deeplabcut model and the desired dataset fieldnames.") ## TODO -- may not need this if we can use the header rows to generate the pose fields
#@produces(group="pose/raw", type="dynamic")
#@param("headerrows", default=[0,1,2], optional=True)

@produces("pose/uncorrected/left/nasal")
@produces("pose/uncorrected/left/pupil")
@produces("pose/uncorrected/left/temporal")
@produces("pose/uncorrected/left/ventral")
@produces("pose/uncorrected/left/dorsal")

@produces("pose/uncorrected/right/nasal")
@produces("pose/uncorrected/right/pupil")
@produces("pose/uncorrected/right/temporal")
@produces("pose/uncorrected/right/ventral")
@produces("pose/uncorrected/right/dorsal")

#def run(base_path, file_pattern, headerrows):
def run(data, params):
    basepath = data["metadata/basepath"]
    file_pattern = params["file_pattern"]#

    dlc_output_file_list = find_files_matching_pattern(basepath, file_pattern, recursive=True)

    if len(dlc_output_file_list) == 0:
        logger.error(f"No dlc output files matching pattern '{file_pattern}' were found in {basepath}")

    #check_scorer_matches_map()
	
    body_part_map = {'N': 'nasal', 'P': 'pupil', 'T': 'temporal', 'U': 'ventral', 'L': 'dorsal'}

    res = {}
    for filename in dlc_output_file_list:
        if "left" in filename.name:
            side = "left"
        elif "right" in filename.name:
            side ="right"
        else:
            logger.warning("side is not right or left")
        #data_runner = pd.read_csv(filename, header=headerrows) ## TODO -- refactor to load_csv_to_df?
        data_runner = pd.read_hdf(filename)
        #header_values = data_runner.columns.values
        #levels = header_values[0] #(scorer, bodyparts, coords) --> assume this will be stable across deeplabcut models
        #for scorer, body_part, coord in header_values[1:]:
        for scorer, body_part, coord in data_runner.columns:
            column_runner = data_runner.xs(body_part, level=data_runner.columns.names[1], axis=1).to_numpy()
            #logger.debug(f"pose/uncorrected/{side}/{body_part_map[body_part]}")
            res[f"pose/uncorrected/{side}/{body_part_map[body_part]}"] = column_runner

    return res ## TODO -- are we setting fields or returning? maybe we just return a list of all the fields produced? Eventually we'll at least be returning messages

		
		

