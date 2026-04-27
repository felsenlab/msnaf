import logging
from pathlib import Path

from flimsy.pipeline.basemodule import *
from flimsy.utils.dat import consolidate_dat_files

logger = logging.getLogger(__name__)

@module(name="loadLabJackDataRig1.0", description="Reads and consolidates sequentially-named LabJack .dat files from a session folder into the results file. Data columns are renamed according to the channel map. Supports the Rig 1.0 .dat acquisition format.")
@requires("metadata/basepath", description="Path to folder containing LabJack .dat files")

@produces("labjack/{channel}/raw")

@param("file_pattern", description="Glob pattern used to find .dat files in the session folder, e.g. 'data_*.dat'. Matched files are read in natural sort order and concatenated.")
@param("channel_map", description="Mapping from raw column names in the .dat files to common field names, e.g. {AIN1: stimulus, FIO6: camera}. Only mapped channels are written to the results file.")
@param("output_filename", default="consolidated_labjack.dat", description=f"Filename for the consolidated .dat file written alongside the session data. Set to null to skip writing. Default: 'consolidated_labjack.dat'.")
def run(data, params):
    basepath = Path(data["metadata/basepath"])
    file_pattern = params["file_pattern"]
    channel_map = params["channel_map"]
    output_filename = params["output_filename"]

    output_path = (basepath / output_filename) if output_filename else None

    labjack_data = consolidate_dat_files(
        basepath,
        pattern=file_pattern,
        output_path=output_path,
    )

    logger.debug(f"Consolidated LabJack data columns: {labjack_data.columns}")

    if output_path is not None:
        logger.info(f"Consolidated LabJack data written to {output_path}")

    res = {}
    res["labjack/timestamps"] = labjack_data["Time"].to_numpy()
    for channel_name, column_name in channel_map.items():
        if channel_name in labjack_data.columns:
            res[f"labjack/{column_name}/raw"] = labjack_data[channel_name].to_numpy()
        else:
            logger.warning(f"Channel '{channel_name}' not found in LabJack data. Available columns: {labjack_data.columns}")

    logger.debug(res)
    return res
