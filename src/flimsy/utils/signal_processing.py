import logging

from sklearn.mixture import GaussianMixture
import numpy as np

logger = logging.getLogger(__name__)



## TODO -- this should warn about poorly separated means
## TODO -- this should warn when not bimodal
## TODO -- this should wran about
def get_state_means(signal):
    gmm = GaussianMixture(n_components=2, covariance_type='full')
    gmm.fit(signal.reshape(-1,1))

    means = gmm.means_.ravel()
    stds  = np.sqrt(gmm.covariances_).ravel()

    return means, stds

def compute_threshold(state_means, state_stdevs, mode):
    if mode == 'bayes':
        return compute_bayes_threshold()
    elif mode == 'mean':
        return state_means.sum()/2

from scipy.ndimage import median_filter
def find_edges(signal, filter=True, mode='mean'):
    if filter == True:
        signal = median_filter(signal, size=5, mode='nearest')
        
    state_means, state_stdevs = get_state_means(signal)
    threshold = compute_threshold(state_means, state_stdevs, mode=mode)
    binarized = signal > threshold

    edge_indices = np.where(np.abs(np.diff(binarized))==1)

    return signal, threshold, edge_indices 
    ## TODO -- we only want to return signal if filter == Tue --> maybe we need to separate that out
    ## TODO -- ok we definitely don't want find_edges doing anything other than finding edges. it should NOT do additional signal processing, it should take a processed signal and maybe the threshold. It should only handle binarziation and the np.where

def filter_dropout(signal, threshold=0.5, max_dropout=5, confirm_low=10):
    """
    Filter sample-level dropouts from a signal that starts low and goes high.
    
    The signal is treated as binary (low/high relative to threshold). Once the
    signal goes high, short dips below threshold are treated as dropouts and
    filled forward. A return to low is only confirmed after `confirm_low`
    consecutive low samples, preventing false endings.
    
    Parameters
    ----------
    signal      : array-like  – raw input signal
    threshold   : float       – value above which the signal is considered "high"
    max_dropout : int         – maximum gap length (samples) to treat as dropout (default 5)
    confirm_low : int         – consecutive low samples required to confirm the
                                signal has truly returned low (default 10)
    
    Returns
    -------
    filtered : np.ndarray – signal with dropouts filled using last good value
    mask     : np.ndarray[bool] – True where a sample was identified as dropout
    """
    signal = np.asarray(signal, dtype=float)
    high = signal >= threshold

    filtered = signal.copy()
    mask = np.zeros(len(signal), dtype=bool)

    in_high_region = False
    dropout_buf = []   # indices of a tentative dropout run
    low_streak = 0     # consecutive lows seen after being high

    for i, h in enumerate(high):
        if not in_high_region:
            if h:
                in_high_region = True
                last_good_val = signal[i]
                dropout_buf = []
                low_streak = 0
        else:
            if h:
                # Genuine high sample — commit any buffered dropout
                if dropout_buf:
                    for j in dropout_buf:
                        filtered[j] = last_good_val
                        mask[j] = True
                    dropout_buf = []
                last_good_val = signal[i]
                low_streak = 0
            else:
                dropout_buf.append(i)
                low_streak += 1

                if low_streak > max_dropout:
                    # Too long to be a dropout — check if it's a real return to low
                    # Use confirm_low to avoid flipping on noise at the trailing edge
                    if low_streak >= confirm_low:
                        # Confirmed return to low — flush buffer as real signal
                        dropout_buf = []
                        in_high_region = False
                    # else: still deciding — keep buffering (will resolve next high or confirm)

    return filtered, mask


"""
matcherv3.py

Match a noisy detector event stream against a reference timing template.

Reference:
    np.ndarray of shape (n,), timestamps in seconds from session start,
    derived from time.monotonic(). Approximate template; the detector is
    the high-precision ground truth.

Rising:
    np.ndarray of shape (m,), sample indices of detector pulse rising edges.
    Converted to seconds via: events_sec = rising / sample_rate.

Scale estimation:
    A scale estimator corrects for clock drift between the reference and
    detector before matching. The scale is applied to the reference.

    Available estimators:
        estimate_scale_span   — fast, uses full span ratio
        estimate_scale_ransac — robust to boundary issues

    Pass as scale_estimator=estimate_scale_span to match().

Matchers (internal):
    _match_linear           — cross-correlation style offset search, scale=1.0
    _match_linear_iterative — 1D ICP: alternates assignment and lstsq refit
                              of scale + offset. Tier 1.
    _match_dp               — minimum cost monotone matching. Tier 2.
    _match_greedy           — internal primitive, not a fallback tier.

Public API:
    match(reference, rising, sample_rate, tolerance, ...)
        -> pairs, times, region_quality, metrics

    fill_missing(times, pairs, reference)
        -> np.ndarray, unmatched slots filled by linear interpolation from
           matched pairs.

    compute_residual_diagnostics(times, pairs, reference, tolerance)
        -> dict of structure test results (Mann-Kendall, Ljung-Box, Slope).

    preprocess(rising, falling, sample_rate, steps)
        -> (rising, falling)
"""

import logging
import numpy as np
from typing import Callable, Optional
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
import pymannkendall as mk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scale estimators
# ---------------------------------------------------------------------------

def estimate_scale_span(
    reference: np.ndarray,
    events_sec: np.ndarray,
) -> float:
    """
    Estimate scale as the ratio of event stream span to reference span.

    Assumes the first and last events correspond to the first and last
    reference slots. Fast and reliable when session boundaries are clean.
    Sensitive to missing events at the start or end of the session.

    Scale > 1.0: detector runs faster than the reference clock.
    Scale < 1.0: detector runs slower.

    Parameters
    ----------
    reference : np.ndarray, shape (n,), seconds
    events_sec : np.ndarray, shape (m,), seconds

    Returns
    -------
    scale : float
        Multiply reference by this to align it to the detector timeline.
    """
    ref_span = reference[-1] - reference[0]
    evt_span = events_sec[-1] - events_sec[0]
    if ref_span <= 0:
        return 1.0
    return float(evt_span / ref_span)


def estimate_scale_ransac(
    reference: np.ndarray,
    events_sec: np.ndarray,
    n_iterations: int = 200,
    inlier_tolerance: float = 1.0,
    min_pairs: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Estimate scale via RANSAC over randomly sampled event pairs.

    Robust to missing or spurious events at session boundaries. For each
    iteration, samples two reference slots and two candidate events, estimates
    scale from their interval ratio, and scores by counting inliers under that
    scale and a median-derived offset.

    Parameters
    ----------
    reference : np.ndarray, shape (n,), seconds
    events_sec : np.ndarray, shape (m,), seconds
    n_iterations : int
        Number of RANSAC trials.
    inlier_tolerance : float
        Seconds. Should be looser than the final matching tolerance —
        this is a coarse scale estimate.
    min_pairs : int
        Minimum inlier count to accept a hypothesis.
    rng : np.random.Generator, optional
        For reproducibility.

    Returns
    -------
    scale : float
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(reference)
    m = len(events_sec)

    best_scale = estimate_scale_span(reference, events_sec)
    best_inliers = 0

    for _ in range(n_iterations):
        i, j = sorted(rng.choice(n, size=2, replace=False))
        ref_interval = reference[j] - reference[i]
        if ref_interval <= 0:
            continue

        ei, ej = sorted(rng.choice(m, size=2, replace=False))
        evt_interval = events_sec[ej] - events_sec[ei]
        if evt_interval <= 0:
            continue

        scale = evt_interval / ref_interval
        offset = events_sec[ei] - scale * reference[i]
        predicted = scale * reference + offset

        inliers = 0
        used = np.zeros(m, dtype=bool)
        for k in range(n):
            diffs = np.abs(events_sec - predicted[k])
            diffs[used] = np.inf
            best_j = int(np.argmin(diffs))
            if diffs[best_j] <= inlier_tolerance:
                inliers += 1
                used[best_j] = True

        if inliers > best_inliers and inliers >= min_pairs:
            best_inliers = inliers
            best_scale = scale

    return float(best_scale)


# ---------------------------------------------------------------------------
# Internal primitives
# ---------------------------------------------------------------------------

def _assign(
    predicted: np.ndarray,
    events_sec: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """
    Nearest-neighbour assignment with a scan cursor.

    For each predicted time in order, finds the nearest unused event within
    tolerance. Returns an index array of shape (n,) where -1 = unmatched.

    The cursor keeps the scan O(m + n) in the common case where predicted
    times and events are both sorted and closely aligned.
    """
    n = len(predicted)
    m = len(events_sec)
    assigned = np.full(n, -1, dtype=int)
    used = np.zeros(m, dtype=bool)
    cursor = 0

    for i in range(n):
        while cursor < m and events_sec[cursor] < predicted[i] - tolerance:
            cursor += 1
        best_j = -1
        best_diff = np.inf
        for j in range(cursor, m):
            if events_sec[j] > predicted[i] + tolerance:
                break
            if not used[j]:
                diff = abs(events_sec[j] - predicted[i])
                if diff < best_diff:
                    best_diff = diff
                    best_j = j
        if best_j >= 0:
            assigned[i] = best_j
            used[best_j] = True

    return assigned


def _build_outputs(
    assigned: np.ndarray,
    events_sec: np.ndarray,
    predicted: np.ndarray,
    tolerance: float,
    method: str,
    offset: float,
    scale: float,
    extra_metrics: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Build the standard (pairs, times, region_quality, metrics) output tuple.

    residuals_per_event is signed (positive = event arrived late relative to
    predicted reference time) and nan where unmatched.
    """
    n = len(assigned)
    m = len(events_sec)

    pairs = assigned >= 0
    times = np.full(n, np.nan)
    times[pairs] = events_sec[assigned[pairs]]

    residuals_per_event = np.full(n, np.nan)
    residuals_per_event[pairs] = times[pairs] - predicted[pairs]
    matched_residuals = residuals_per_event[pairs]

    metrics = {
        "method": method,
        "offset": float(offset),
        "scale": float(scale),
        "tolerance": float(tolerance),
        "match_rate": float(np.sum(pairs) / n),
        "n_matched": int(np.sum(pairs)),
        "n_reference": n,
        "n_events": m,
        "mean_residual": float(np.mean(matched_residuals)) if len(matched_residuals) else np.nan,
        "std_residual": float(np.std(matched_residuals)) if len(matched_residuals) else np.nan,
        "max_abs_residual": float(np.max(np.abs(matched_residuals))) if len(matched_residuals) else np.nan,
        "tolerance_utilization": (
            float(np.mean(np.abs(matched_residuals)) / tolerance)
            if len(matched_residuals) else np.nan
        ),
        "residuals_per_event": residuals_per_event,
    }

    if extra_metrics:
        metrics.update(extra_metrics)

    return pairs, times, _region_quality(pairs), metrics


def _region_quality(
    pairs: np.ndarray,
    bad_threshold: int = 3,
) -> np.ndarray:
    """
    Label each reference slot 0 (good) or 1 (bad).

    A slot is labeled bad if it is unmatched and part of a consecutive run
    of >= bad_threshold unmatched slots. Isolated unmatched slots within
    otherwise well-matched regions are labeled good.
    """
    n = len(pairs)
    quality = np.zeros(n, dtype=int)
    i = 0
    while i < n:
        if not pairs[i]:
            j = i
            while j < n and not pairs[j]:
                j += 1
            if (j - i) >= bad_threshold:
                quality[i:j] = 1
            i = j
        else:
            i += 1
    return quality


# ---------------------------------------------------------------------------
# Internal matchers
# ---------------------------------------------------------------------------

def _match_linear(
    reference: np.ndarray,
    events_sec: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Cross-correlation style matcher. Scale fixed at 1.0.

    Tries anchoring every reference slot to every event to generate candidate
    offsets. Scores each by match count, keeping the best. Only evaluates
    offsets where the predicted window overlaps the observed event stream.

    Assumes scale ≈ 1.0. Will degrade when clock drift is significant relative
    to tolerance × session_duration. Use _match_linear_iterative instead when
    drift is known or suspected.

    Complexity: O((m + n) * n).
    """
    # Two sets of candidate offsets:
    # a) anchor each reference slot to the first event — handles sessions that
    #    start partway into the reference window.
    # b) anchor reference[0] to each event — handles the common case.
    # Together these cover boundary offsets without an O(m*n) outer product.
    offsets = np.unique(np.concatenate([
        events_sec[0] - reference,
        events_sec - reference[0],
    ]))

    best_offset = float(events_sec[0] - reference[0])
    best_score = -1
    best_assigned = np.full(len(reference), -1, dtype=int)

    for offset in offsets:
        predicted = reference + offset
        # Skip offsets with no overlap between predicted window and event stream
        if predicted[-1] < events_sec[0] - tolerance * 2:
            continue
        if predicted[0] > events_sec[-1] + tolerance * 2:
            continue
        assigned = _assign(predicted, events_sec, tolerance)
        score = int(np.sum(assigned >= 0))
        if score > best_score:
            best_score = score
            best_offset = float(offset)
            best_assigned = assigned.copy()

    predicted = reference + best_offset
    return _build_outputs(
        best_assigned, events_sec, predicted, tolerance,
        method="linear", offset=best_offset, scale=1.0,
    )


def _match_linear_iterative(
    reference: np.ndarray,
    events_sec: np.ndarray,
    tolerance: float,
    n_iterations: int = 5,
    tol_schedule: Optional[list] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    1D temporal registration via iterative assignment and lstsq refit.

    Analogous to Iterative Closest Point (ICP): alternates between assigning
    each reference slot to its nearest event (under the current linear model)
    and refitting scale + offset from those correspondences via least squares.

    A geometrically decaying tolerance schedule allows early iterations to cast
    a wide net for initial alignment, with later iterations refining under the
    requested tolerance. Converges in 2-3 iterations for well-behaved sessions.

    Initialised with estimate_scale_span for scale and a first-event anchor
    for offset, providing a coarse but reliable starting point.

    Parameters
    ----------
    reference : np.ndarray, shape (n,), seconds (pre-scaled by scale_estimator)
    events_sec : np.ndarray, shape (m,), seconds
    tolerance : float
        Final matching tolerance in seconds. Used as the last step of the
        tolerance schedule.
    n_iterations : int
        Number of assignment-refit iterations.
    tol_schedule : list of float, optional
        Tolerance at each iteration. Defaults to geometrically decaying from
        5 * tolerance to tolerance over n_iterations steps.
    """
    if tol_schedule is None:
        tol_schedule = list(np.geomspace(tolerance * 5, tolerance, n_iterations))

    scale = estimate_scale_span(reference, events_sec)
    offset = float(events_sec[0] - scale * reference[0])
    assigned = np.full(len(reference), -1, dtype=int)

    for tol in tol_schedule:
        predicted = scale * reference + offset
        assigned = _assign(predicted, events_sec, tol)

        matched_ref = np.where(assigned >= 0)[0]
        matched_evt = assigned[matched_ref]

        if len(matched_ref) < 2:
            continue

        # Least-squares refit: events_sec ~ scale * reference + offset
        A = np.column_stack([reference[matched_ref], np.ones(len(matched_ref))])
        b = events_sec[matched_evt]
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        scale, offset = float(result[0]), float(result[1])

    predicted = scale * reference + offset
    return _build_outputs(
        assigned, events_sec, predicted, tolerance,
        method="linear_iterative", offset=offset, scale=scale,
        extra_metrics={"n_iterations": n_iterations},
    )


def _match_greedy(
    reference: np.ndarray,
    events_sec: np.ndarray,
    tolerance: float,
    offset: Optional[float] = None,
    scale: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Sequential nearest-neighbour matcher seeded by a prior offset and scale.

    Internal primitive — not used as a fallback tier. Makes irrevocable local
    decisions; best used after a good global fit has been established.
    """
    if offset is None:
        offset = float(events_sec[0] - reference[0])
    if scale is None:
        scale = 1.0

    predicted = scale * reference + offset
    assigned = _assign(predicted, events_sec, tolerance)

    return _build_outputs(
        assigned, events_sec, predicted, tolerance,
        method="greedy", offset=offset, scale=scale,
    )


def _match_dp(
    reference: np.ndarray,
    events_sec: np.ndarray,
    tolerance: float,
    mismatch_penalty: Optional[float] = None,
    offset: Optional[float] = None,
    scale: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Minimum cost monotone matching via dynamic programming.

    Finds the globally optimal assignment of detector events to reference
    slots minimising total timing deviation, subject to:
      - Monotonicity: assignments preserve temporal order
      - Each event used at most once
      - Unmatched reference slots incur mismatch_penalty
      - Spurious events (insertions) are skipped at zero cost

    mismatch_penalty defaults to tolerance: skipping a slot costs the same
    as a match at the edge of the tolerance window.

    offset and scale seed the predicted reference times, typically from
    the Tier 1 iterative fit.
    """
    n = len(reference)
    m = len(events_sec)

    if offset is None:
        offset = float(events_sec[0] - reference[0])
    if scale is None:
        scale = 1.0
    if mismatch_penalty is None:
        mismatch_penalty = tolerance

    predicted = scale * reference + offset

    INF = float("inf")
    dp = np.full((n + 1, m + 1), INF)
    dp[0, :] = 0.0

    # back[i][j]:
    #   > 0  — matched to event back[i,j] (1-based)
    #     0  — reference slot i skipped
    #    -1  — event j skipped (carry from j-1)
    back = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(1, n + 1):
        for j in range(0, m + 1):
            # Option 1: skip reference slot i
            cost = dp[i - 1, j] + mismatch_penalty
            if cost < dp[i, j]:
                dp[i, j] = cost
                back[i, j] = 0

            # Option 2: match reference slot i to event j
            if j > 0:
                diff = abs(events_sec[j - 1] - predicted[i - 1])
                if diff <= tolerance:
                    cost = dp[i - 1, j - 1] + diff
                    if cost < dp[i, j]:
                        dp[i, j] = cost
                        back[i, j] = j  # 1-based

            # Option 3: skip event j (spurious insertion)
            if j > 0 and dp[i, j - 1] < dp[i, j]:
                dp[i, j] = dp[i, j - 1]
                back[i, j] = -1

    # Traceback
    assigned = np.full(n, -1, dtype=int)
    j = m
    for i in range(n, 0, -1):
        b = back[i, j]
        if b > 0:
            assigned[i - 1] = b - 1
            j = b - 1
        elif b == 0:
            pass  # ref slot skipped
        else:
            while j > 0 and back[i, j] == -1:
                j -= 1
            b2 = back[i, j]
            if b2 > 0:
                assigned[i - 1] = b2 - 1
                j = b2 - 1

    return _build_outputs(
        assigned, events_sec, predicted, tolerance,
        method="dp", offset=offset, scale=scale,
        extra_metrics={
            "mismatch_penalty": float(mismatch_penalty),
            "total_cost": float(dp[n, m]),
        },
    )


# ---------------------------------------------------------------------------
# Residual diagnostics
# ---------------------------------------------------------------------------

def compute_residual_diagnostics(
    times: np.ndarray,
    pairs: np.ndarray,
    reference: np.ndarray,
    tolerance: float,
) -> dict:
    """
    Test matched residuals for systematic structure using three independent
    statistical tests, each sensitive to a different failure mode.

    Tests are run on the residuals of matched pairs only:
        residuals[i] = times[i] - predicted[i]

    where predicted times come from a lstsq refit of scale + offset over
    matched pairs (same fit used by fill_missing).

    Tests
    -----
    Mann-Kendall:
        Nonparametric test for monotone trend (linear or nonlinear drift).
        Sensitive to cases where the linear model does not fully capture
        clock drift. p < 0.05 → warn.

    Ljung-Box:
        Tests for autocorrelation structure across lags 1-5. Sensitive to
        cyclic or oscillatory residuals (e.g. temperature cycling, AC
        interference). p < 0.05 → warn. Lag timescale reported in seconds
        using mean inter-event spacing from reference.

    Slope F-test:
        Linear regression of residuals vs reference time. p < 0.05 and
        |slope| > tolerance / session_duration → warn. Slope is reported
        in ms/s for interpretability.

    Parameters
    ----------
    times : np.ndarray, shape (n,), seconds, nan where unmatched
    pairs : np.ndarray, shape (n,), bool
    reference : np.ndarray, shape (n,), seconds
    tolerance : float, seconds

    Returns
    -------
    dict with keys:
        diag_n_matched          : int
        diag_mk_pvalue          : float  Mann-Kendall p-value
        diag_mk_significant     : bool
        diag_lb_pvalue          : float  Ljung-Box minimum p-value over lags 1-5
        diag_lb_significant     : bool
        diag_lb_lag_interval_s  : float  mean inter-event spacing in seconds
        diag_slope_pvalue       : float  F-test p-value for slope
        diag_slope_magnitude_ms_per_s : float  |slope| in ms/s
        diag_slope_significant  : bool
    """
    n_matched = int(np.sum(pairs))
    if n_matched < 4:
        logger.warning(
            "compute_residual_diagnostics: only %d matched pairs, "
            "structure tests are unreliable at small n.", n_matched
        )
        return {
            "diag_n_matched": n_matched,
            "diag_mk_pvalue": np.nan,
            "diag_mk_significant": False,
            "diag_lb_pvalue": np.nan,
            "diag_lb_significant": False,
            "diag_lb_lag_interval_s": np.nan,
            "diag_slope_pvalue": np.nan,
            "diag_slope_magnitude_ms_per_s": np.nan,
            "diag_slope_significant": False,
        }

    t_matched = reference[pairs]
    times_matched = times[pairs]

    # Refit scale + offset from matched pairs — same as fill_missing
    A = np.column_stack([t_matched, np.ones(n_matched)])
    result, _, _, _ = np.linalg.lstsq(A, times_matched, rcond=None)
    scale_fit, offset_fit = float(result[0]), float(result[1])
    predicted_matched = scale_fit * t_matched + offset_fit
    residuals = times_matched - predicted_matched

    session_duration = float(reference[-1] - reference[0])
    slope_threshold = tolerance / session_duration if session_duration > 0 else np.inf

    # Mann-Kendall
    mk_result = mk.original_test(residuals)
    mk_pvalue = float(mk_result.p)
    mk_significant = mk_pvalue < 0.05
    if mk_significant:
        logger.warning(
            "compute_residual_diagnostics: Mann-Kendall trend test significant "
            "(p=%.4f). Residuals show monotone structure — linear model may not "
            "fully capture clock drift. fill_missing interpolation may be unreliable.",
            mk_pvalue
        )

    # Ljung-Box
    lb_result = acorr_ljungbox(residuals, lags=5, return_df=True)
    lb_pvalue = float(lb_result["lb_pvalue"].min())
    lb_significant = lb_pvalue < 0.05
    lag_interval_s = float(np.diff(t_matched).mean()) if n_matched > 1 else np.nan
    if lb_significant:
        logger.warning(
            "compute_residual_diagnostics: Ljung-Box autocorrelation test "
            "significant (p=%.4f, lag interval ~%.2fs). Residuals show cyclic "
            "or persistent structure. fill_missing interpolation may be unreliable.",
            lb_pvalue, lag_interval_s
        )

    # Slope F-test
    slope_result = stats.linregress(t_matched, residuals)
    slope_pvalue = float(slope_result.pvalue)
    slope_magnitude = abs(float(slope_result.slope))
    slope_significant = slope_pvalue < 0.05 and slope_magnitude > slope_threshold
    if slope_significant:
        logger.warning(
            "compute_residual_diagnostics: Residual slope significant "
            "(p=%.4f, slope=%.4f ms/s, threshold=%.4f ms/s). Remaining linear "
            "drift not captured by the alignment. fill_missing interpolation "
            "will accumulate ~%.1f ms of error over the session.",
            slope_pvalue,
            slope_magnitude * 1000,
            slope_threshold * 1000,
            slope_magnitude * session_duration * 1000,
        )

    return {
        "diag_n_matched": n_matched,
        "diag_mk_pvalue": mk_pvalue,
        "diag_mk_significant": mk_significant,
        "diag_lb_pvalue": lb_pvalue,
        "diag_lb_significant": lb_significant,
        "diag_lb_lag_interval_s": lag_interval_s,
        "diag_slope_pvalue": slope_pvalue,
        "diag_slope_magnitude_ms_per_s": float(slope_magnitude * 1000),
        "diag_slope_significant": slope_significant,
    }


# ---------------------------------------------------------------------------
# fill_missing
# ---------------------------------------------------------------------------

def fill_missing(
    times: np.ndarray,
    pairs: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """
    Fill unmatched reference slots with reference-interpolated timestamps.

    Fits a linear model (scale + offset) to the matched pairs via least
    squares, then uses it to predict timestamps for unmatched slots. Returns
    a new array — does not modify times in place.

    This provides detector-timeline timestamps for unmatched slots based on
    the best available alignment. Accuracy depends on the quality of the
    match; call compute_residual_diagnostics first to assess reliability.

    Parameters
    ----------
    times : np.ndarray, shape (n,), seconds, nan where unmatched
    pairs : np.ndarray, shape (n,), bool
    reference : np.ndarray, shape (n,), seconds

    Returns
    -------
    filled : np.ndarray, shape (n,), no nans
    """
    n_matched = int(np.sum(pairs))
    if n_matched < 2:
        raise ValueError(
            f"fill_missing requires at least 2 matched pairs, got {n_matched}."
        )

    t_matched = reference[pairs]
    times_matched = times[pairs]

    A = np.column_stack([t_matched, np.ones(n_matched)])
    result, _, _, _ = np.linalg.lstsq(A, times_matched, rcond=None)
    scale_fit, offset_fit = float(result[0]), float(result[1])

    filled = times.copy()
    unmatched = ~pairs
    filled[unmatched] = scale_fit * reference[unmatched] + offset_fit

    return filled


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(
    rising: np.ndarray,
    falling: np.ndarray,
    sample_rate: float,
    steps: list,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a sequence of preprocessor functions to the event arrays.

    Each step is a callable:
        step(rising, falling, sample_rate) -> (rising, falling)

    Steps are applied in order; each receives the output of the previous.
    """
    for step in steps:
        rising, falling = step(rising, falling, sample_rate)
    return rising, falling


def filter_min_duration(min_samples: int):
    """Keep events whose duration (falling - rising) >= min_samples."""
    def _filter(rising, falling, sample_rate):
        mask = (falling - rising) >= min_samples
        return rising[mask], falling[mask]
    return _filter


def filter_max_duration(max_samples: int):
    """Keep events whose duration (falling - rising) <= max_samples."""
    def _filter(rising, falling, sample_rate):
        mask = (falling - rising) <= max_samples
        return rising[mask], falling[mask]
    return _filter


def filter_min_interval(min_samples: int):
    """Keep events where interval to previous rising edge >= min_samples.
    The first event is always kept."""
    def _filter(rising, falling, sample_rate):
        if len(rising) < 2:
            return rising, falling
        mask = np.concatenate([[True], np.diff(rising) >= min_samples])
        return rising[mask], falling[mask]
    return _filter


def filter_max_interval(max_samples: int):
    """Keep events where interval to previous rising edge <= max_samples.
    The first event is always kept."""
    def _filter(rising, falling, sample_rate):
        if len(rising) < 2:
            return rising, falling
        mask = np.concatenate([[True], np.diff(rising) <= max_samples])
        return rising[mask], falling[mask]
    return _filter


# ---------------------------------------------------------------------------
# Public matcher
# ---------------------------------------------------------------------------

QUALITY_THRESHOLD = 0.9


def match(
    reference: np.ndarray,
    rising: np.ndarray,
    sample_rate: float,
    tolerance: float,
    fallback: bool = False,
    scale_estimator: Optional[Callable] = None,
    compute_diagnostics: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Match a noisy detector event stream to a reference timing template.

    The reference is an approximate template in reference-clock time. The
    detector (rising edges) is the high-precision ground truth. If a
    scale_estimator is provided, the reference is rescaled to the detector
    timeline before matching.

    Tier 1: _match_linear_iterative — 1D ICP, jointly estimates scale and
            offset via alternating assignment and lstsq refit.
    Tier 2: _match_dp — globally optimal monotone assignment given the Tier 1
            scale and offset, handles insertions and deletions explicitly.

    Parameters
    ----------
    reference : np.ndarray, shape (n,)
        Timestamps in seconds from session start (time.monotonic() - start).
    rising : np.ndarray, shape (m,)
        Sample indices of detector pulse rising edges.
    sample_rate : float
        Samples per second.
    tolerance : float
        Maximum allowed timing deviation in seconds for a match to be accepted.
    fallback : bool
        If True, attempt Tier 2 (DP) if Tier 1 fails to meet quality threshold.
        If False, raise ValueError on Tier 1 failure.
    scale_estimator : callable, optional
        Function (reference, events_sec) -> float. Applied to reference before
        matching. Options: estimate_scale_span, estimate_scale_ransac.
        Recommended when clock drift between reference and detector is expected.
    compute_diagnostics : bool
        If True, run compute_residual_diagnostics on the result and merge into
        metrics. Adds Mann-Kendall, Ljung-Box, and slope test results.

    Returns
    -------
    pairs : np.ndarray, shape (n,), bool
        True where a detector event was matched to the reference slot.
    times : np.ndarray, shape (n,), float
        Matched event times in seconds from session start. nan where unmatched.
    region_quality : np.ndarray, shape (n,), int
        0 = good, 1 = bad (run of >= 3 consecutive unmatched slots).
    metrics : dict
        Always contains: method, offset, scale, tolerance, match_rate,
        n_matched, n_reference, n_events, mean_residual, std_residual,
        max_abs_residual, tolerance_utilization, residuals_per_event,
        pre_scale, final_scale.
        If compute_diagnostics=True, also contains diag_* keys.
    """
    events_sec = rising / sample_rate

    # Pre-scale the reference to the detector timeline
    if scale_estimator is not None:
        pre_scale = scale_estimator(reference, events_sec)
        reference = reference * pre_scale
        logger.debug("match: pre_scale from estimator = %.6f", pre_scale)
    else:
        pre_scale = 1.0

    # Tier 1: iterative linear registration
    pairs, times, region_quality, metrics = _match_linear_iterative(
        reference, events_sec, tolerance
    )

    final_scale = metrics["scale"]
    logger.debug("match: final_scale from Tier 1 = %.6f", final_scale)

    if pre_scale != 1.0:
        relative_deviation = abs(pre_scale - final_scale) / pre_scale
        if relative_deviation > 0.01:
            logger.warning(
                "match: pre_scale (%.6f) and final_scale (%.6f) deviate by %.2f%%. "
                "Scale estimator and iterative fit disagree — check for boundary "
                "issues or poor initialisation.",
                pre_scale, final_scale, relative_deviation * 100
            )

    metrics["pre_scale"] = pre_scale
    metrics["final_scale"] = final_scale

    if metrics["match_rate"] >= QUALITY_THRESHOLD:
        if compute_diagnostics:
            metrics.update(
                compute_residual_diagnostics(times, pairs, reference, tolerance)
            )
        return pairs, times, region_quality, metrics

    if not fallback:
        raise ValueError(
            f"Tier 1 (linear_iterative) achieved match_rate="
            f"{metrics['match_rate']:.3f} (threshold {QUALITY_THRESHOLD}). "
            f"mean_residual={metrics['mean_residual']:.4f}s, "
            f"max_abs_residual={metrics['max_abs_residual']:.4f}s, "
            f"tolerance={tolerance}s. "
            f"Re-run with fallback=True to attempt DP matcher, or check "
            f"tolerance and scale_estimator settings."
        )

    # Tier 2: DP seeded with Tier 1 scale and offset
    pairs, times, region_quality, metrics = _match_dp(
        reference, events_sec, tolerance,
        offset=metrics.get("offset"),
        scale=metrics.get("scale"),
    )

    metrics["pre_scale"] = pre_scale
    metrics["final_scale"] = metrics["scale"]

    if metrics["match_rate"] >= QUALITY_THRESHOLD:
        if compute_diagnostics:
            metrics.update(
                compute_residual_diagnostics(times, pairs, reference, tolerance)
            )
        return pairs, times, region_quality, metrics

    raise ValueError(
        f"All matchers failed. match_rate={metrics['match_rate']:.3f}. "
        f"mean_residual={metrics['mean_residual']:.4f}s, "
        f"max_abs_residual={metrics['max_abs_residual']:.4f}s, "
        f"tolerance={tolerance}s. "
        f"Consider loosening tolerance or inspecting the session data."
    )