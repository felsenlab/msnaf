"""
parseDriftingGratingMetadata.py
--------------------------------
Updated drifting-grating stimulus parser.  Replaces the original
two-pass photologic filter with a median filter, and replaces the
hardcoded file list with natural-sort auto-discovery.  Mismatch
recovery logic is carried over faithfully from the legacy module.

Config example
--------------
parseDriftingGratingMetadata:
  stimulus_channel: labjack/stimulus/raw
  median_filter_size: 5
  inter_block_interval_threshold: 16000   # samples
  timing_mismatch_threshold: 0.1          # seconds
  metadata_dir: videos                    # relative to basepath
  file_pattern: "driftingGratingMetadata-*.txt"
  n_header_lines: 5
  header_fields:
    spatial_frequency: 0
    velocity: 1
    orientation: 2
    baseline_contrast: 3
  column_map:
    event_id: 0
    motion_direction: 1
    probe_contrast: 2
    probe_phase: 3
    timestamp: 4
"""

import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from flimsy.pipeline.basemodule import *
from flimsy.utils.ioer import find_files_matching_pattern
from flimsy.utils.dat import _natural_sort_key

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: parse a single metadata .txt file
# ---------------------------------------------------------------------------

def _parse_metadata_file(filepath, n_header_lines, header_fields, column_map):
    """
    Parse one drifting-grating metadata file.

    Header lines are plain prose of the form
        "Field name: value (units)"
    so we split on ':' and strip the units parenthetical.

    Returns
    -------
    header_values : dict  field_name -> float
    data          : np.ndarray  shape (n_events, n_columns)
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    header_values = {}
    for field_name, line_idx in header_fields.items():
        raw = lines[line_idx].split(":")[1].split("(")[0].strip()
        try:
            header_values[field_name] = float(raw)
        except ValueError:
            header_values[field_name] = raw

    data = np.genfromtxt(filepath, skip_header=n_header_lines, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    return header_values, data


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

@module(
    name="parseDriftingGratingMetadata",
    description=(
        "Drifting-grating stimulus parser. Uses median filtering for "
        "signal cleaning and natural-sort auto-discovery of metadata files. "
        "Carries over timing-based mismatch recovery from the legacy module. "
        "Produces identical h5 outputs to parseDriftingGratingMetadataLegacy "
        "for direct comparison."
    ),
)
@requires("metadata/basepath", description="Path to session folder")
@requires("labjack/timestamps", description="Wall-clock time for every labjack sample (1-D array)")
@requires("labjack/stimulus/raw")

#@param("stimulus_channel", description="H5 field containing the raw stimulus TTL signal, e.g. labjack/stimulus/raw")
@param("median_filter_size", default=5, description="Kernel size for median filter used to clean the TTL signal. Must be odd.")
@param("inter_block_interval_threshold", default=16000, description="Minimum inter-pulse gap in samples that separates stimulus blocks")
@param("timing_mismatch_threshold", default=0.1, description="Inter-event timing difference in seconds above which a pulse is considered missing")
#@param("metadata_dir", default="videos", description="Subdirectory of basepath containing the metadata files")
@param("file_pattern", default="driftingGratingMetadata-*.txt", description="Glob pattern for discovering metadata files. Files are sorted in natural order.")
@param("n_header_lines", default=5, description="Number of non-CSV header lines at the top of each metadata file")
@param("header_fields", description="Mapping of field name to 0-based header line index, e.g. {orientation: 2, velocity: 1}")
@param("column_map", description="Mapping of field name to 0-based CSV column index, e.g. {event_id: 0, motion_direction: 1, timestamp: 4}")

@produces("stimuli/dg/probe/timestamps")
@produces("stimuli/dg/probe/motion")
@produces("stimuli/dg/grating/timestamps")
@produces("stimuli/dg/grating/motion")
@produces("stimuli/dg/grating/contrast")
@produces("stimuli/dg/grating/velocity")
@produces("stimuli/dg/motion/timestamps")
@produces("stimuli/dg/iti/timestamps")

def run(data, params):
    basepath      = Path(data["metadata/basepath"])
    lj_timestamps = data["labjack/timestamps"]
    signal        = data["labjack/stimulus/raw"]

    filter_size             = params["median_filter_size"]
    inter_block_threshold   = params["inter_block_interval_threshold"]
    timing_mismatch_thresh  = params["timing_mismatch_threshold"]
    #metadata_dir            = params["metadata_dir"]
    file_pattern            = params["file_pattern"]
    n_header_lines          = params["n_header_lines"]
    header_fields           = params["header_fields"]
    column_map              = params["column_map"]

    col_motion    = column_map["motion_direction"]
    col_timestamp = column_map["timestamp"]

    # ------------------------------------------------------------------
    # 1. Filter signal and find pulse edges
    # ------------------------------------------------------------------
    filtered = median_filter(signal, size=filter_size)

    i_pulses      = np.where(np.diff(filtered) > 0.5)[0]
    i_pulses_fall = np.where(np.diff(filtered) < -0.5)[0]

    if len(i_pulses) == 0:
        logger.error("No rising edges found in stimulus signal after filtering. Check stimulus_channel and median_filter_size.")
        return {}

    if len(i_pulses) != len(i_pulses_fall):
        logger.warning(
            f"Asymmetric edge count: {len(i_pulses)} rising, "
            f"{len(i_pulses_fall)} falling. Signal may be truncated."
        )

    # Block boundaries: inter-pulse gaps larger than threshold
    i_intervals       = np.where(np.diff(i_pulses) > inter_block_threshold)[0]
    i_intervals2      = i_pulses[i_intervals]
    pulse_timestamps  = lj_timestamps[i_pulses]
    interval_timestamps = lj_timestamps[i_intervals2] + 3

    # ------------------------------------------------------------------
    # 2. Discover and sort metadata files
    # ------------------------------------------------------------------
    #parent_dir     = basepath / metadata_dir
    resolved_files = sorted(
        find_files_matching_pattern(basepath, file_pattern, recursive=True),
        key=_natural_sort_key,
    )

    if len(resolved_files) == 0:
        logger.error(f"No metadata files matching '{file_pattern}' found in {basepath}")
        return {}

    n_blocks_expected = interval_timestamps.shape[0] + 1
    if len(resolved_files) != n_blocks_expected:
        logger.error(
            f"Discovered {len(resolved_files)} metadata file(s) but expected "
            f"{n_blocks_expected} block(s). Check file_pattern and "
            f"inter_block_interval_threshold."
        )
        return {}

    logger.info(f"Found {len(resolved_files)} metadata file(s) in {basepath}")

    # ------------------------------------------------------------------
    # 3. Build metadataHolder
    #
    # Column layout (matches legacy module):
    #   0  event_id
    #   1  motion_direction
    #   2  probe_contrast
    #   3  probe_phase
    #   4  event_timestamp (from metadata file)
    #   5  block_type  (0 = DG, reserved for multi-protocol sessions)
    #   6  trial_type  (reserved)
    #   7  orientation
    #   8  velocity
    #   9  baseline_contrast
    # ------------------------------------------------------------------
    N_COLS = 5
    metadata_holder = np.full((len(i_pulses) + 100, N_COLS), np.nan)
    event_index = 0

    for file_index, filepath in enumerate(resolved_files):
        header_values, block_data = _parse_metadata_file(
            filepath, n_header_lines, header_fields, column_map
        )

        orientation  = header_values.get("orientation",       np.nan)
        velocity     = header_values.get("velocity",          np.nan)
        contrast     = header_values.get("baseline_contrast", np.nan)
        block_length = block_data.shape[0]

        # Determine which pulses belong to this block
        if file_index == 0:
            block_mask = pulse_timestamps < interval_timestamps[0]
        elif file_index == len(resolved_files) - 1:
            block_mask = pulse_timestamps > interval_timestamps[-1]
        else:
            block_mask = (
                (pulse_timestamps > interval_timestamps[file_index - 1]) &
                (pulse_timestamps < interval_timestamps[file_index])
            )

        n_pulses_observed    = block_mask.sum()
        pulse_count_mismatch = n_pulses_observed != block_length

        if pulse_count_mismatch:
            logger.warning(
                f"{filepath.name}: pulse count mismatch "
                f"(observed {n_pulses_observed}, expected {block_length}). "
                f"Attempting timing-based recovery."
            )
            this_block_pulses = pulse_timestamps[block_mask]
            this_block_diffs  = np.diff(this_block_pulses)
            from_file_diffs   = np.diff(block_data[:, col_timestamp])

            prev_pulse_missing   = False
            missing_pulse_offset = 0
            for i in range(from_file_diffs.shape[0]):
                if prev_pulse_missing:
                    metadata_holder[event_index + i, :] = np.nan
                    prev_pulse_missing = False
                else:
                    observed_idx = i - missing_pulse_offset
                    if observed_idx < len(this_block_diffs):
                        this_diff = from_file_diffs[i] - this_block_diffs[observed_idx]
                    else:
                        this_diff = timing_mismatch_thresh + 1  # force NaN

                    if this_diff > timing_mismatch_thresh:
                        prev_pulse_missing   = True
                        missing_pulse_offset += 1

                    metadata_holder[event_index + i, :5] = block_data[i, :5]
                    metadata_holder[event_index + i,  5] = 0
                    metadata_holder[event_index + i,  7] = orientation
                    metadata_holder[event_index + i,  8] = velocity
                    metadata_holder[event_index + i,  9] = contrast

            event_index += block_length

        else:
            metadata_holder[event_index: event_index + block_length, :5] = block_data
            metadata_holder[event_index: event_index + block_length,  5] = 0
            metadata_holder[event_index: event_index + block_length,  7] = orientation
            metadata_holder[event_index: event_index + block_length,  8] = velocity
            metadata_holder[event_index: event_index + block_length,  9] = contrast
            event_index += block_length

    # ------------------------------------------------------------------
    # 4. Extract outputs from metadataHolder
    # ------------------------------------------------------------------
    def _timestamps_for_event(event_id, block_type=None):
        mask = metadata_holder[:, 0] == event_id
        if block_type is not None:
            mask &= metadata_holder[:, 5] == block_type
        row_indices   = np.where(mask)[0]
        pulse_indices = i_pulses[row_indices]
        return lj_timestamps[pulse_indices]

    # DG probe (event_id=3, block_type=0)
    dg_mask        = (metadata_holder[:, 0] == 3) & (metadata_holder[:, 5] == 0)
    dg_row_indices = np.where(dg_mask)[0]
    probe_timestamps = lj_timestamps[i_pulses[dg_row_indices]]
    probe_motion     = metadata_holder[dg_row_indices, col_motion]

    grating_row_idx  = np.where(metadata_holder[:, 0] == 1)[0]
    grating_timestamps = lj_timestamps[i_pulses[grating_row_idx]]
    grating_motion     = metadata_holder[grating_row_idx, col_motion]
    grating_contrast   = metadata_holder[grating_row_idx, 9]
    grating_velocity   = metadata_holder[grating_row_idx, 8]

    motion_timestamps = _timestamps_for_event(2)
    iti_timestamps    = _timestamps_for_event(4)

    n_missing = np.isnan(metadata_holder[:event_index, 0]).sum()
    if n_missing > 0:
        logger.warning(f"{n_missing} event(s) could not be matched to a pulse and were set to NaN.")

    return {
        "stimuli/dg/probe/timestamps":   probe_timestamps,
        "stimuli/dg/probe/motion":        probe_motion,
        "stimuli/dg/grating/timestamps": grating_timestamps,
        "stimuli/dg/grating/motion":     grating_motion,
        "stimuli/dg/grating/contrast":   grating_contrast,
        "stimuli/dg/grating/velocity":   grating_velocity,
        "stimuli/dg/motion/timestamps":  motion_timestamps,
        "stimuli/dg/iti/timestamps":     iti_timestamps,
    }
