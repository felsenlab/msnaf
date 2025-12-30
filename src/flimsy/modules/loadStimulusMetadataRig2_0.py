import pandas as pd

from flimsy.pipeline.basemodule import *
from flimsy.utils.ioer import find_files_matching_pattern

logger = logging.getLogger(__name__)

## NOTE: this is technically true for both API versions, but the below is mostly relevant for the 1.0 API. Per-frame diode pulses makes missed pulses a lot easier to deal with
## TODO -- there are two jobs here --> one is to load the metadata, the other is to timestamp it
#          Photodiode can be unreliable, so we first check to see if we drop pulses. Because this the timing of events is random, we can't really infer timing?
#          WHY ARE WE ADDING EVENTS TO THE METADATA HOLDER? SURELY THIS IS ONLY REQUIRED IF len(events) < len(pulses), but if we are dropping pulses then we should have more events than pulses, which adding events just makes worse????
#          We then timestamp each event, based on the pulses that they were matched to in step 1
## TODO -- some remaining decision making to do here -- how do we break all this up: we have loading, parsing (separating events), and timestamping (matching rows to lj events). Is this one, two, or three modules?

@module(name="", description="")
@requires("base_path", description="Path to session folder") ## TODO -- this was probably already provided earlier, should we get this from the h5?
@requires("file_pattern", description="List of patterns that will be used to identify stimulus metadata output files. The wildcard operator * matches any number of characters (including none) in a file name, so a pattern like file_pattern='*grating*.csv' will select all files that contain 'grating' and end with '.csv', such as 'driftingGratingWithRandomProbe-1.csv' and 'driftingGratingWithRandomProbe-1-flip_interval.csv'; similarly, ? matches exactly one character.")
@produces("/stimuli/")
@param("n_expected_files", default=2)
def run(base_path, file_patterns, n_expected_files):
	## we expect two files for each stimulus: the metadata and the flip intervals. I think we load these both at the same time. Do we want to load all stimuli at the same time or separately? How do we do that?
	res = {pattern:{} for pattern in file_patterns}
	for pattern in file_patterns:
		metadata_files = sorted(find_files_matching_pattern(base_path, pattern))

		if len(metadata_files == n_expected_files):
			metadata = pd.read_csv(metadata_files[0])
			intervals = pd.read_csv(metadata_files[1])
			res[pattern]["metadata"] = metadata
			res[pattern]["intervals"]
			

