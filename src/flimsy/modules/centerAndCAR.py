import numpy as np

from flimsy.pipeline.basemodule import *

logger = logging.getLogger(__name__)

@module(name="centerAndCAR", 
        description="Centers pose estimates such that the landmark centroid has a mean position of [0, 0] and performs common average referencing (CAR) to remove global eye motion. CAR subtracts the median centroid of the nasal, temporal, dorsal, and ventral landmarks on a per-frame basis.")

@param("filter_landmarks", description="Boolean representing whether or not to median filter the landmark position to remove pixel-level jitter. Recommended value is True")

@requires("pose/interpolated/left/nasal")
@requires("pose/interpolated/left/pupil")
@requires("pose/interpolated/left/temporal")
@requires("pose/interpolated/left/ventral")
@requires("pose/interpolated/left/dorsal")

@requires("pose/interpolated/right/nasal")
@requires("pose/interpolated/right/pupil")
@requires("pose/interpolated/right/temporal")
@requires("pose/interpolated/right/ventral")
@requires("pose/interpolated/right/dorsal")


@produces("pose/centered/left/nasal")
@produces("pose/centered/left/pupil")
@produces("pose/centered/left/temporal")
@produces("pose/centered/left/ventral")
@produces("pose/centered/left/dorsal")

@produces("pose/centered/right/nasal")
@produces("pose/centered/right/pupil")
@produces("pose/centered/right/temporal")
@produces("pose/centered/right/ventral")
@produces("pose/centered/right/dorsal")


@produces("pose/corrected/left/nasal", description="Pose estimate after common average referencing")
@produces("pose/corrected/left/pupil", description="Pose estimate after common average referencing")
@produces("pose/corrected/left/temporal", description="Pose estimate after common average referencing")
@produces("pose/corrected/left/ventral", description="Pose estimate after common average referencing")
@produces("pose/corrected/left/dorsal", description="Pose estimate after common average referencing")

@produces("pose/corrected/right/nasal", description="Pose estimate after common average referencing")
@produces("pose/corrected/right/pupil", description="Pose estimate after common average referencing")
@produces("pose/corrected/right/temporal", description="Pose estimate after common average referencing")
@produces("pose/corrected/right/ventral", description="Pose estimate after common average referencing")
@produces("pose/corrected/right/dorsal", description="Pose estimate after common average referencing")

@produces("pose/car/right", description="Estimated global motion of the eye using the median landmark centroid. NOTE: this is a PRE-ROTATION estimate.")
@produces("pose/car/left", description="Estimated global motion of the eye using the median landmark centroid. NOTE: this is a PRE-ROTATION estimate.")

def run(data, params):
    filter_landmarks = params["filter_landmarks"]

    res = {}
    for side in ["left", "right"]:
        nasal_xy = data[f"pose/interpolated/{side}/nasal"]
        temporal_xy = data[f"pose/interpolated/{side}/temporal"]
        dorsal_xy = data[f"pose/interpolated/{side}/dorsal"]
        ventral_xy = data[f"pose/interpolated/{side}/ventral"]
        pupil_xy = data[f"pose/interpolated/{side}/pupil"]

        car = compute_car(nasal_xy, temporal_xy, dorsal_xy, ventral_xy)

        centered_nasal_xy = nasal_xy - car.mean(axis=0)
        centered_temporal_xy = temporal_xy - car.mean(axis=0)
        centered_ventral_xy = ventral_xy - car.mean(axis=0)
        centered_dorsal_xy = dorsal_xy - car.mean(axis=0)
        centered_pupil_xy = pupil_xy - car.mean(axis=0)

        res[f"pose/centered/{side}/nasal"] = centered_nasal_xy
        res[f"pose/centered/{side}/temporal"] = centered_temporal_xy
        res[f"pose/centered/{side}/dorsal"] = centered_dorsal_xy
        res[f"pose/centered/{side}/ventral"] = centered_ventral_xy
        res[f"pose/centered/{side}/pupil"] = centered_pupil_xy

        res[f"pose/corrected/{side}/nasal"] = nasal_xy - car
        res[f"pose/corrected/{side}/temporal"] = temporal_xy - car
        res[f"pose/corrected/{side}/dorsal"] = dorsal_xy - car
        res[f"pose/corrected/{side}/ventral"] = ventral_xy - car
        res[f"pose/corrected/{side}/pupil"] = pupil_xy - car

        res[f"pose/car/{side}"] = car

    return res

def compute_car(nasal_xy, temporal_xy, dorsal_xy, ventral_xy):
    """
    Compute robust per-frame translation estimate (CAR) using median of landmarks.
    
    Returns:
        car: (n_frames, 2) median across landmarks per frame
    """
    stacked = np.stack([nasal_xy, temporal_xy,
                        dorsal_xy, ventral_xy], axis=0)
    car = np.nanmedian(stacked, axis=0)
    return car