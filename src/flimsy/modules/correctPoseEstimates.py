import logging

from flimsy.pipeline.registry import module

logger = logging.getLogger(__name__)

@module(name='correctPoseEstimates')
def run():
    # blank low confidence estiamtes --> maybe zscore not raw threshold?
    # rotate / project onto nasal-temporal axis and center
    # find and blank dropped frames
    # interpolate gaps (low confidence + dropped frames)
    ## TODO -- previous note that nans were being introduced at high-confidence samples
    ## TODO -- do we want to median filter to remove high frequency noise?
    masked_pose = mask_low_confidence_samples()
    corrected_pose = rotate_and_center()
    imputed_pose = interpolate_gaps()
    if smooth:
        smooth_signal()

def mask_low_confidence_samples(pose, confidence)
    threshold = '' ## TODO -- figure out how to threshold nicely --> should it be percentile or scalar
    confidence_mask = confidence < threshold
    pose[confidence_mask] = np.nan
    return pose

def rotate_and_center():
    ## TODO -- understand the math better. maybe be able to avoid mean subtraction if we use single origin
    mean_nasal_pos = np.nanmean(nasal_pose, axis=0)
    mean_temporal_pos = np.nanmean(nasal_pose, axis=0)
    mean_dorsal_pos = np.nanmean(nasal_pose, axis=0)
    mean_ventral_pos = np.nanmean(nasal_pose, axis=0)

    mean_subtracted =  
    horizontal_projection = np.dot(, )

def interpolate_gaps():
    pass