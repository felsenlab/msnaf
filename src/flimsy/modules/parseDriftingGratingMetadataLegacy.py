"""
parseDriftingGratingMetadataLegacy.py
--------------------------------------
Faithful port of the NR1KO _calculatePulseIntervalTimestamps /
_createMetadataFileList / _processDriftingGratingProtocol /
_parseMetadataHolder pipeline into the flimsy module framework.

filterPulsesFromPhotologicDevice is reproduced here verbatim so the
legacy module is entirely self-contained and its behaviour can be
verified against the original without any external changes.

Config example
--------------
parseDriftingGratingMetadataLegacy:
  stimulus_channel: labjack/stimulus/raw
  inter_block_interval_threshold: 16000   # samples between blocks
  timing_mismatch_threshold: 0.1          # seconds
  metadata_dir: videos                    # relative to basepath
  file_list:
    - driftingGratingMetadata-0.txt
    - driftingGratingMetadata-1.txt
    - ...
  n_header_lines: 5
  header_fields:
    # line index (0-based) -> field name
    spatial_frequency: 0
    velocity: 1
    orientation: 2
    baseline_contrast: 3
  column_map:
    # column index (0-based) within the CSV section -> field name
    event_id: 0
    motion_direction: 1
    probe_contrast: 2
    probe_phase: 3
    timestamp: 4
"""

import logging
from pathlib import Path

import numpy as np

from flimsy.pipeline.basemodule import *

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy filter — reproduced verbatim from myphdlib.general.labjack
# ---------------------------------------------------------------------------

def _filter_pulses_from_photologic_device(
    signal,
    minimumPulseWidthInSeconds=0.015,
    samplingRate=1000,
):
    """
    Original two-pass edge-interval filter.  Reproduced verbatim so that
    the legacy module produces bit-identical results to the original code.
    """
    threshold = round(minimumPulseWidthInSeconds * samplingRate, 2)
    mutated = np.copy(signal)

    # Pass 1 — fill short low intervals (intra-pulse dropout)
    while True:
        deltaState = np.diff(mutated)
        edgeIndices = np.where(abs(deltaState) > 0.5)[0]
        intervals = np.hstack([
            edgeIndices[0:-1].reshape(-1, 1),
            edgeIndices[1:  ].reshape(-1, 1),
        ])
        for firstEdgeIndex, secondEdgeIndex in intervals:
            dt = secondEdgeIndex - firstEdgeIndex
            if dt < threshold:
                mutated[firstEdgeIndex + 1: secondEdgeIndex + 1] = 1
        break

    # Pass 2 — zero short high intervals (noise spikes)
    while True:
        deltaState = np.diff(mutated)
        edgeIndices = np.where(abs(deltaState) > 0.5)[0]
        intervals = np.hstack([
            edgeIndices[0:-1].reshape(-1, 1),
            edgeIndices[1:  ].reshape(-1, 1),
        ])
        for firstEdgeIndex, secondEdgeIndex in intervals:
            dt = secondEdgeIndex - firstEdgeIndex
            if dt < threshold:
                mutated[firstEdgeIndex + 1: secondEdgeIndex + 1] = 0
        break

    deltaState = np.diff(mutated)
    edgeIndices = np.where(abs(deltaState) > 0.5)[0]
    if np.all(np.diff(edgeIndices) > threshold):
        return mutated
    else:
        raise Exception("Pulse filtering failed")


# ---------------------------------------------------------------------------
# Helper: parse a single metadata .txt file
# ---------------------------------------------------------------------------

def _parse_metadata_file(filepath, n_header_lines, header_fields, column_map):
    """
    Parse one drifting-grating metadata file.

    Returns
    -------
    header_values : dict
        Mapping of field name -> float value, one entry per header_fields entry.
    data : np.ndarray
        Shape (n_events, n_columns) CSV section.
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
    name="parseDriftingGratingMetadataLegacy",
    description=(
        "Legacy port of the NR1KO drifting-grating stimulus parser. "
        "Uses the original two-pass photologic filter and a config-supplied "
        "ordered file list. Produces the same h5 outputs as the original "
        "_parseMetadataHolder for direct comparison with the updated module."
    ),
)
@requires("metadata/basepath", description="Path to session folder")
@requires("labjack/timestamps", description="Wall-clock time for every labjack sample (1-D array)")
@param("stimulus_channel", description="H5 field containing the raw stimulus TTL signal, e.g. labjack/stimulus/raw")
@param("inter_block_interval_threshold", default=16000, description="Minimum inter-pulse gap in samples that separates stimulus blocks")
@param("timing_mismatch_threshold", default=0.1, description="Inter-event timing difference in seconds above which a pulse is considered missing")
@param("metadata_dir", default="videos", description="Subdirectory of basepath containing the metadata files")
@param("file_list", description="Ordered list of metadata filenames. Order determines block assignment.")
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
    basepath        = Path(data["metadata/basepath"])
    lj_timestamps   = data["labjack/timestamps"]
    signal          = data[params["stimulus_channel"]]

    inter_block_threshold   = params["inter_block_interval_threshold"]
    timing_mismatch_thresh  = params["timing_mismatch_threshold"]
    metadata_dir            = params["metadata_dir"]
    file_list               = params["file_list"]
    n_header_lines          = params["n_header_lines"]
    header_fields           = params["header_fields"]
    column_map              = params["column_map"]

    col_event_id        = column_map["event_id"]
    col_motion          = column_map["motion_direction"]
    col_timestamp       = column_map["timestamp"]

    # ------------------------------------------------------------------
    # 1. Filter signal and find pulse edges
    #    (_calculatePulseIntervalTimestamps)
    # ------------------------------------------------------------------
    filtered = _filter_pulses_from_photologic_device(signal)

    i_pulses      = np.where(np.diff(filtered) > 0.5)[0]
    i_pulses_fall = np.where(np.diff(filtered) < -0.5)[0]
    pulse_durations = np.subtract(i_pulses_fall, i_pulses)

    # Block boundaries: inter-pulse gaps larger than threshold
    i_intervals  = np.where(np.diff(i_pulses) > inter_block_threshold)[0]
    i_intervals2 = i_pulses[i_intervals]
    pulse_timestamps    = lj_timestamps[i_pulses]
    interval_timestamps = lj_timestamps[i_intervals2] + 3

    # ------------------------------------------------------------------
    # 2. Resolve file list
    #    (_createMetadataFileList)
    # ------------------------------------------------------------------
    parent_dir = basepath / metadata_dir
    resolved_files = [parent_dir / fname for fname in file_list]

    missing = [f for f in resolved_files if not f.exists()]
    if missing:
        logger.error(f"Missing metadata files: {missing}")
        return {}

    n_blocks_expected = interval_timestamps.shape[0] + 1
    if len(resolved_files) != n_blocks_expected:
        logger.error(
            f"File list length ({len(resolved_files)}) does not match "
            f"expected number of blocks ({n_blocks_expected}). "
            f"Check file_list param and inter_block_interval_threshold."
        )
        return {}

    # ------------------------------------------------------------------
    # 3. Build metadataHolder
    #    (_processDriftingGratingProtocol)
    #
    # Column layout (matches NR1KO original):
    #   0  event_id
    #   1  motion_direction
    #   2  probe_contrast
    #   3  probe_phase
    #   4  event_timestamp (from metadata file)
    #   5  block_type  (0 = DG, reserved slot for FS compatibility)
    #   6  trial_type  (reserved)
    #   7  orientation
    #   8  velocity
    #   9  contrast (baseline)
    # ------------------------------------------------------------------
    N_COLS = 10
    metadata_holder = np.full((len(i_pulses) + 100, N_COLS), np.nan)
    event_index = 0

    for file_index, filepath in enumerate(resolved_files):
        if filepath.suffix != ".txt":
            # Non-DG file in list — skip without advancing event_index
            # (extend here if SN/FS blocks need to reserve slots)
            continue

        header_values, block_data = _parse_metadata_file(
            filepath, n_header_lines, header_fields, column_map
        )

        orientation = header_values.get("orientation", np.nan)
        velocity    = header_values.get("velocity",    np.nan)
        contrast    = header_values.get("baseline_contrast", np.nan)
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

        n_pulses_observed = block_mask.sum()
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

            prev_pulse_missing  = False
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
                        prev_pulse_missing = True
                        missing_pulse_offset += 1

                    metadata_holder[event_index + i, :5]  = block_data[i, :5]
                    metadata_holder[event_index + i,  5]  = 0
                    metadata_holder[event_index + i,  7]  = orientation
                    metadata_holder[event_index + i,  8]  = velocity
                    metadata_holder[event_index + i,  9]  = contrast

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
    #    (_parseMetadataHolder)
    # ------------------------------------------------------------------
    def _timestamps_for_event(event_id, block_type=None):
        """Return lj_timestamps for rows matching event_id (and optionally block_type)."""
        mask = metadata_holder[:, 0] == event_id
        if block_type is not None:
            mask &= metadata_holder[:, 5] == block_type
        row_indices = np.where(mask)[0]
        pulse_indices = i_pulses[row_indices]
        return lj_timestamps[pulse_indices]

    # DG probe (event_id=3, block_type=0)
    dg_mask        = (metadata_holder[:, 0] == 3) & (metadata_holder[:, 5] == 0)
    dg_row_indices = np.where(dg_mask)[0]
    probe_timestamps = lj_timestamps[i_pulses[dg_row_indices]]
    probe_motion     = metadata_holder[dg_row_indices, col_motion]

    grating_timestamps = _timestamps_for_event(1)
    grating_row_idx    = np.where(metadata_holder[:, 0] == 1)[0]
    grating_motion     = metadata_holder[grating_row_idx, col_motion]
    grating_contrast   = metadata_holder[grating_row_idx, 9]
    grating_velocity   = metadata_holder[grating_row_idx, 8]

    motion_timestamps = _timestamps_for_event(2)
    iti_timestamps    = _timestamps_for_event(4)

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
