import numpy as np

from flimsy.pipeline.basemodule import *
from flimsy.utils.saccades import smooth_signal

logger = logging.getLogger(__name__)

@module(name="rotatePose", description="Rotates pose estimates so that nasal-temporal axis is horizontal and dosal-ventral axis is vertical")
@param('flip_dv_axis', default=True, description="Boolean representing whether or not to flip the dorsal-ventral axis to correct for camera being upside down")
@param('smoothing_strength', default=1, description="Boolean representing whether or not to flip the dorsal-ventral axis to correct for camera being upside down")

@requires("pose/corrected/left/nasal")
@requires("pose/corrected/left/pupil")
@requires("pose/corrected/left/temporal")
@requires("pose/corrected/left/ventral")
@requires("pose/corrected/left/dorsal")

@requires("pose/corrected/right/nasal")
@requires("pose/corrected/right/pupil")
@requires("pose/corrected/right/temporal")
@requires("pose/corrected/right/ventral")
@requires("pose/corrected/right/dorsal")

@produces("pose/rotated/left/pupil")
@produces("pose/rotated/right/pupil")

def run(data, params):
    flip_dv_axis = params["flip_dv_axis"]
    smoothing_strength = params["smoothing_strength"]

    res = {}
    for side in ["left", "right"]:
        nasal_xy = data[f"pose/corrected/{side}/nasal"]
        temporal_xy = data[f"pose/corrected/{side}/temporal"]
        dorsal_xy = data[f"pose/corrected/{side}/dorsal"]
        ventral_xy = data[f"pose/corrected/{side}/ventral"]
        pupil_xy = data[f"pose/corrected/{side}/pupil"]

        R, nt_hat, ul_hat = compute_rotation_matrix(nasal_xy, temporal_xy, dorsal_xy, ventral_xy, flip_vertical=flip_dv_axis)
        rotated_pupil = rotate_points(pupil_xy, R)
        res[f"pose/rotated/{side}/pupil"] = rotated_pupil
        logger.warning(rotated_pupil.shape)

        # smoothed = np.copy(rotated_pupil)
        # smoothed[:,0] = medfilt(smoothed[:,0], kernel_size=3)
        # smoothed[:,1] = medfilt(smoothed[:,1], kernel_size=3)

        res[f"pose/smoothed/{side}"] = smooth_signal(rotated_pupil, smoothing_strength, method='gaussian')  #(3,1), method='median'    np.round(150*0.003, 2), method='gaussian'
        res[f"pose/rotation/{side}/nt_vector"] = nt_hat
        res[f"pose/rotation/{side}/dv_vector"] = ul_hat
        res[f"pose/rotation/{side}/matrix"] = R

    return res


def compute_rotation_matrix(nasal_xy, temporal_xy, dorsal_xy, ventral_xy, flip_vertical=False):
    """
    Compute a fixed, robust NT/UL rotation matrix from landmarks.
    
    Returns:
        R: 2x2 rotation matrix (rows are axes)
        nt_hat, ul_hat: unit vectors defining axes
    """
    # Global, robust median-based axis estimate
    median_n = np.median(nasal_xy, axis=0)
    median_t = np.median(temporal_xy, axis=0)
    median_u = np.median(dorsal_xy, axis=0)
    median_l = np.median(ventral_xy, axis=0)

    nt = median_t - median_n
    ul = median_u - median_l if flip_vertical else median_l - median_u

    # Orthonormalize
    nt_hat = nt / np.linalg.norm(nt)
    ul = ul - np.dot(ul, nt_hat) * nt_hat
    ul_hat = ul / np.linalg.norm(ul)

    R = np.stack([nt_hat, ul_hat], axis=0)
    return R, nt_hat, ul_hat

def rotate_points(points, R, origin=None):
    """
    Rotate points into a new basis defined by R.
    
    points: (n,2)
    R: 2x2 rotation matrix (rows are axes)
    origin: optional, (2,) vector to subtract before rotation
    """
    if origin is not None:
        points = points - origin
    rotated = points @ R.T
    return rotated