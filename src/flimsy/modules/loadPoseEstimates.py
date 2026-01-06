import pandas as pd

from flimsy.pipeline.basemodule import *
from flimsy.utils.ioer import find_files_matching_pattern
from flimsy.utils.validation import check_scorer_matches_map

logger = logging.getLogger(__name__)

@module(name="loadPoseEstimates", description="This module will find all deeplabcut output files matching the provided pattern in the session folder, regardless of folder structure (it will search all subfolders for the file pattern)")
@requires("base_path", description="Path to session folder") ## TODO -- this was probably already provided earlier, should we get this from the h5?
@param("file_pattern", description="String that will be used to identify deeplabcut output files. The pipeline is designed to load the h5 output, rather than csvs. The wildcard operator * matches any number of characters (including none) in a file name, so a pattern like video_pattern='*DLC*.csv' will select all files that contain 'DLC' and end with '.csv', such as '20251217_unitME_session009_leftCam-0000DLC_resnet101_sacnetJan29shuffle1_1030000.csv' or '20251217_unitME_session009_rightCam-0000DLC_resnet101_sacnetJan29shuffle1_1030000.csv'; similarly, ? matches exactly one character.")
#@param("mapfile", description="Path to a configuration yaml file that defines the columns output by the deeplabcut model and the desired dataset fieldnames.") ## TODO -- may not need this if we can use the header rows to generate the pose fields
#@produces(group="pose/raw", type="dynamic")
#@param("headerrows", default=[0,1,2], optional=True)
@produces("pose/uncorrected/{side}")
def run(base_path, file_pattern, headerrows):
	dlc_output_file_list = find_files_matching_pattern(base_path, file_pattern, recursive=True)

	#check_scorer_matches_map()

	res = {}
	for filename in dlc_output_file_list:
		print(filename)
		#data_runner = pd.read_csv(filename, header=headerrows) ## TODO -- refactor to load_csv_to_df?
		data_runner = pd.read_hdf(filename)
		#header_values = data_runner.columns.values
		#levels = header_values[0] #(scorer, bodyparts, coords) --> assume this will be stable across deeplabcut models
		res[filename] = {}
		#for scorer, body_part, coord in header_values[1:]:
		for scorer, body_part, coord in data_runner.columns:
			column_runner = data_runner.xs(body_part, level=data_runner.columns.names[1], axis=1).to_numpy()
			res[filename]['_'.join([body_part,coord]) ] = column_runner

	return res ## TODO -- are we setting fields or returning? maybe we just return a list of all the fields produced? Eventually we'll at least be returning messages

		
		

