import pandas as pd

from flimsy.pipeline.basemodule import *

logger = logging.getLogger(__name__)

@module(name="loadLabJackDataRig2.0", description="Reads and saves labjack signal into results file based on a channel map. Supports labjack format")
@requires("filepath", description="Path to labjack data csv file")
@produces("/labjack/raw/*")
@produces("/labjack/raw/*")
## TODO -- 
def run(filepath, channel_map):
    labjack_data = pd.read_csv(filepath)
    print(labjack_data.columns)

    res = {}
    for channel_name, column_name in channel_map.items():
        if column_name in labjack_data:
            res[channel_name] = labjack_data[column_name] 

    return res




## TODO -- how to mark optional outputs? what we output kinda depends on the channel map...
# @module(name='parseLabJackMetadata', requires=['labjack_datapath', 'channel_map'], produces=['frameTimestamps', 'cameraTimestamps'])
# def parse_labjack_metadata(labjack_datapath, channel_map):
#     lj_data = pd.read_csv(labjack_datapath)

#     camera_edge_indices = parse_camera_signal(lj_data['channel_map']['camera'])
#     frame_edge_indices = parse_frame_signal(lj_data['channel_map']['photologic'])

#     camera_timestamps = timestamp_from_sample(lj_sample_timestamps, camera_edge_indices)
#     frame_timestamps = timestamp_from_sample(lj_sample_timestamps, frame_edge_indices)



    
    











    