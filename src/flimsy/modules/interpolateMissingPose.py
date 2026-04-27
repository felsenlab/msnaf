import numpy as np

from flimsy.pipeline.basemodule import *
from flimsy.utils.saccades import mask_low_confidence_samples, insert_frames, interpolate_gaps, identify_dropped_frames

logger = logging.getLogger(__name__)

@module(name="interpolateMissingPose", description="Masks low confidence pose estimates, identifies missing frames, and interpolates the masked pose positions")

@param("confidence_threshold", default=1.0)
@param("framerate", default=150)
@param("framedrop_threshold", default=1.5)

@requires("pose/uncorrected/{side}/{bodypart}")
@requires("frames/{side}/intervals")

@param("fieldnames")

@produces("pose/masked/{side}/{bodypart}")
@produces("pose/interopolated/{side}/{bodypart}")

# @requires("pose/uncorrected/left/nasal")
# @requires("pose/uncorrected/left/pupil")
# @requires("pose/uncorrected/left/temporal")
# @requires("pose/uncorrected/left/ventral")
# @requires("pose/uncorrected/left/dorsal")
# @requires("pose/uncorrected/right/nasal")
# @requires("pose/uncorrected/right/pupil")
# @requires("pose/uncorrected/right/temporal")
# @requires("pose/uncorrected/right/ventral")
# @requires("pose/uncorrected/right/dorsal")

# @requires("frames/left/intervals")
# @requires("frames/right/intervals")

# @produces("pose/masked/left/nasal")
# @produces("pose/masked/left/pupil")
# @produces("pose/masked/left/temporal")
# @produces("pose/masked/left/ventral")
# @produces("pose/masked/left/dorsal")

# @produces("pose/interpolated/left/nasal")
# @produces("pose/interpolated/left/pupil")
# @produces("pose/interpolated/left/temporal")
# @produces("pose/interpolated/left/ventral")
# @produces("pose/interpolated/left/dorsal")

# @produces("pose/masked/right/nasal")
# @produces("pose/masked/right/pupil")
# @produces("pose/masked/right/temporal")
# @produces("pose/masked/right/ventral")
# @produces("pose/masked/right/dorsal")

# @produces("pose/interpolated/right/nasal")
# @produces("pose/interpolated/right/pupil")
# @produces("pose/interpolated/right/temporal")
# @produces("pose/interpolated/right/ventral")
# @produces("pose/interpolated/right/dorsal")

def run(data, params):
    confidence_threshold = params["confidence_threshold"]
    framerate = params["framerate"]
    framedrop_threshold = params["framedrop_threshold"]

    res = {}
    for side in params["fieldnames"]["side"]:
        pupil_pose = data[f"pose/uncorrected/{side}/pupil"]
        nasal_pose = data[f"pose/uncorrected/{side}/nasal"]
        temporal_pose = data[f"pose/uncorrected/{side}/temporal"]
        dorsal_pose = data[f"pose/uncorrected/{side}/dorsal"]
        ventral_pose = data[f"pose/uncorrected/{side}/ventral"]

        frame_intervals = data[f"frames/{side}/intervals"]

        masked_pupil_pose = mask_low_confidence_samples(pupil_pose, threshold=confidence_threshold)
        masked_nasal_pose = mask_low_confidence_samples(nasal_pose, threshold=confidence_threshold)
        masked_temporal_pose = mask_low_confidence_samples(temporal_pose, threshold=confidence_threshold)
        masked_dorsal_pose = mask_low_confidence_samples(dorsal_pose, threshold=confidence_threshold)
        masked_ventral_pose = mask_low_confidence_samples(ventral_pose, threshold=confidence_threshold)

        dropped_frames, frames_to_insert = identify_dropped_frames(frame_intervals, len(masked_pupil_pose), framerate, framedrop_threshold)
        
        logger.debug(f"Inserting frames: {side} Pupil")
        indexcorrected_pupil_pose = insert_frames(masked_pupil_pose, frames_to_insert)
        logger.debug(f"Inserting frames: {side} Nasal")
        indexcorrected_nasal_pose = insert_frames(masked_nasal_pose, frames_to_insert)
        logger.debug(f"Inserting frames: {side} Temporal")
        indexcorrected_temporal_pose = insert_frames(masked_temporal_pose, frames_to_insert)
        logger.debug(f"Inserting frames: {side} Dorsal")
        indexcorrected_dorsal_pose = insert_frames(masked_dorsal_pose, frames_to_insert)
        logger.debug(f"Inserting frames: {side} Ventral")
        indexcorrected_ventral_pose = insert_frames(masked_ventral_pose, frames_to_insert)
        
        logger.debug(f"Interpolating gaps: {side} Pupil")
        imputed_pose_pupil = interpolate_gaps(indexcorrected_pupil_pose)

        logger.debug(f"Interpolating gaps: {side} Nasal")
        imputed_pose_nasal = interpolate_gaps(indexcorrected_nasal_pose)

        logger.debug(f"Interpolating gaps: {side} Temporal")
        imputed_pose_temporal = interpolate_gaps(indexcorrected_temporal_pose)

        logger.debug(f"Interpolating gaps: {side} Dorsal")
        imputed_pose_dorsal = interpolate_gaps(indexcorrected_dorsal_pose)

        logger.debug(f"Interpolating gaps: {side} Ventral")
        imputed_pose_ventral = interpolate_gaps(indexcorrected_ventral_pose)


        res[f"pose/masked/{side}/pupil"] = indexcorrected_pupil_pose[:,[0,1]]
        res[f"pose/masked/{side}/pupil"] = indexcorrected_nasal_pose[:,[0,1]]
        res[f"pose/masked/{side}/pupil"] = indexcorrected_temporal_pose[:,[0,1]]
        res[f"pose/masked/{side}/pupil"] = indexcorrected_dorsal_pose[:,[0,1]]
        res[f"pose/masked/{side}/pupil"] = indexcorrected_ventral_pose[:,[0,1]]

        res[f"pose/interpolated/{side}/pupil"] = imputed_pose_pupil[:,[0,1]]
        res[f"pose/interpolated/{side}/nasal"] = imputed_pose_nasal[:,[0,1]]
        res[f"pose/interpolated/{side}/temporal"] = imputed_pose_temporal[:,[0,1]]
        res[f"pose/interpolated/{side}/dorsal"] = imputed_pose_dorsal[:,[0,1]]
        res[f"pose/interpolated/{side}/ventral"] = imputed_pose_ventral[:,[0,1]]

    return res