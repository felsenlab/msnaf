import pandas as pd

from flimsy.pipeline.basemodule import *
from flimsy.utils.ioer import find_files_matching_pattern

logger = logging.getLogger(__name__)

@module(name="loadLabJackDataRig2.0", description="Reads and saves labjack signal into results file based on a channel map. Supports labjack format")
@requires("/metadata/basepath/", description="Path to folder containing labjack data csv file")
@produces("/labjack/{channel}/raw") ## TODO -- how to mark optional outputs? what we output kinda depends on the channel map...
@param("file_pattern", description="String that will be used to identify the labjack data file. The wildcard operator * matches any number of characters (including none) in a file name, so a pattern like file_pattern='*labjack*.csv' will select all files that contain 'labjack' and end with '.csv', such as 'labjack_data.csv'; similarly, ? matches exactly one character.")
## TODO -- I don't like have a default value here, but it feels annoying to make people put it in the config. 
@param("channel_map", default={"AIN1":"stimulus", "FIO6":"camera", "FI06":"camera"}, description="YAML config file containing column name: common name pairs") ## TODO -- maybe get rid of this cause we can just take fieldnames in the later modules (NOTE: this is made annoying by the O/0 confusion)
def run(basepath, channel_map, file_pattern):
    filepath = find_files_matching_pattern(basepath, file_pattern)
    labjack_data = pd.read_csv(filepath)
    print(labjack_data.columns)

    res = {}
    for channel_name, column_name in channel_map.items():
        if column_name in labjack_data:
            res[channel_name] = labjack_data[column_name] 

    return res





    
    











    