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


from typing import Optional


# ---------------------------------------------------------------------------
# Internal matchers
# ---------------------------------------------------------------------------

def _match_linear(
    reference: np.ndarray,
    events_sec: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Tier 1: Linear fit (offset + scale) via least-squares, then assign
    each reference slot to the nearest unassigned event within tolerance.

    Finds the global offset and scale that best align the reference comb
    to the event stream, then scores matches. A boundary penalty discourages
    solutions that push the fit off the edges of the event stream.

    Returns
    -------
    pairs : np.ndarray, shape (n,), bool
    times : np.ndarray, shape (n,), float seconds, nan if unmatched
    region_quality : np.ndarray, shape (n,), int, 0=good 1=bad
    metrics : dict
    """
    n = len(reference)
    m = len(events_sec)

    # Fit offset + scale: events_sec ~ offset + scale * reference
    # Use least-squares over all candidate pairs would be expensive;
    # instead, try all (reference[0] -> event[j]) anchors and score each.
    # Scale is estimated from the ratio of total spans.
    ref_span = reference[-1] - reference[0]
    evt_span = events_sec[-1] - events_sec[0]
    scale = evt_span / ref_span if ref_span > 0 else 1.0

    # Candidate offsets: anchor reference[0] to each event
    predicted_starts = events_sec - scale * reference[0]

    best_offset = None
    best_scale = scale
    best_score = -1
    best_pairs = None

    for offset in predicted_starts:
        predicted = offset + scale * reference
        # Boundary penalty: skip if predicted range falls outside event stream
        if predicted[0] < events_sec[0] - tolerance * 2:
            continue
        if predicted[-1] > events_sec[-1] + tolerance * 2:
            continue

        # Score: count how many reference slots have an event within tolerance
        assigned = np.full(n, -1, dtype=int)
        used = np.zeros(m, dtype=bool)
        for i in range(n):
            diffs = np.abs(events_sec - predicted[i])
            diffs[used] = np.inf
            j = np.argmin(diffs)
            if diffs[j] <= tolerance:
                assigned[i] = j
                used[j] = True

        score = np.sum(assigned >= 0)
        if score > best_score:
            best_score = score
            best_offset = offset
            best_pairs = assigned.copy()

    # Build outputs
    pairs = best_pairs >= 0
    times = np.full(n, np.nan)
    times[pairs] = events_sec[best_pairs[pairs]]

    residuals = np.abs(times[pairs] - (best_offset + best_scale * reference[pairs]))
    region_quality = _region_quality(pairs)

    metrics = {
        "method": "linear",
        "offset": best_offset,
        "scale": best_scale,
        "match_rate": np.sum(pairs) / n,
        "mean_residual": float(np.mean(residuals)) if len(residuals) else np.nan,
        "std_residual": float(np.std(residuals)) if len(residuals) else np.nan,
        "n_matched": int(np.sum(pairs)),
        "n_reference": n,
        "n_events": m,
    }

    return pairs, times, region_quality, metrics


def _match_greedy(
    reference: np.ndarray,
    events_sec: np.ndarray,
    tolerance: float,
    offset: Optional[float] = None,
    scale: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Tier 2: Sequential greedy matching, informed by a prior linear fit.

    Walks reference slots in order. For each slot, looks for the nearest
    unused event within tolerance of the predicted time. If no event is
    found, the slot is marked unmatched and the search continues.

    If offset/scale are provided (from Tier 1), uses them to predict
    expected times. Otherwise falls back to raw reference times.

    Returns
    -------
    pairs : np.ndarray, shape (n,), bool
    times : np.ndarray, shape (n,), float seconds, nan if unmatched
    region_quality : np.ndarray, shape (n,), int, 0=good 1=bad
    metrics : dict
    """
    n = len(reference)
    m = len(events_sec)

    if offset is None:
        offset = events_sec[0] - reference[0]
    if scale is None:
        scale = 1.0

    predicted = offset + scale * reference
    assigned = np.full(n, -1, dtype=int)
    used = np.zeros(m, dtype=bool)

    # Maintain a search cursor to avoid O(nm) scan
    cursor = 0
    for i in range(n):
        # Advance cursor to first event that could be within tolerance
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

    pairs = assigned >= 0
    times = np.full(n, np.nan)
    times[pairs] = events_sec[assigned[pairs]]

    residuals = np.abs(times[pairs] - predicted[pairs])
    region_quality = _region_quality(pairs)

    metrics = {
        "method": "greedy",
        "offset": offset,
        "scale": scale,
        "match_rate": float(np.sum(pairs) / n),
        "mean_residual": float(np.mean(residuals)) if len(residuals) else np.nan,
        "std_residual": float(np.std(residuals)) if len(residuals) else np.nan,
        "n_matched": int(np.sum(pairs)),
        "n_reference": n,
        "n_events": m,
    }

    return pairs, times, region_quality, metrics


def _match_dp(
    reference: np.ndarray,
    events_sec: np.ndarray,
    tolerance: float,
    mismatch_penalty: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Tier 3: Minimum cost monotone matching via dynamic programming.

    Finds the assignment of detector events to reference slots that
    minimizes total timing deviation, subject to:
      - Monotonicity: assignments must preserve order
      - Each event assigned to at most one slot
      - Unmatched slots incur a fixed mismatch penalty

    mismatch_penalty defaults to tolerance, meaning skipping a slot
    costs the same as a match at the edge of the tolerance window.

    Returns
    -------
    pairs : np.ndarray, shape (n,), bool
    times : np.ndarray, shape (n,), float seconds, nan if unmatched
    region_quality : np.ndarray, shape (n,), int, 0=good 1=bad
    metrics : dict
    """
    n = len(reference)
    m = len(events_sec)

    if mismatch_penalty is None:
        mismatch_penalty = tolerance

    # Offset reference to event stream start for fair comparison
    offset = events_sec[0] - reference[0]
    ref_shifted = reference + offset

    # dp[i][j] = min cost to match first i reference slots using first j events
    # We use 1-indexed arrays for convenience
    INF = float("inf")
    dp = np.full((n + 1, m + 1), INF)
    dp[0, :] = 0.0  # zero cost to match zero reference slots

    # Backpointer: -1 = skip ref slot, j = matched to event j
    back = np.full((n + 1, m + 1), -1, dtype=int)

    for i in range(1, n + 1):
        for j in range(0, m + 1):
            # Option 1: skip reference slot i (unmatched)
            skip_cost = dp[i - 1, j] + mismatch_penalty
            if skip_cost < dp[i, j]:
                dp[i, j] = skip_cost
                back[i, j] = -1  # skipped

            # Option 2: match reference slot i to event j (1-indexed)
            if j > 0:
                diff = abs(events_sec[j - 1] - ref_shifted[i - 1])
                if diff <= tolerance:
                    match_cost = dp[i - 1, j - 1] + diff
                    if match_cost < dp[i, j]:
                        dp[i, j] = match_cost
                        back[i, j] = j  # matched to event j (1-indexed)

            # Option 3: skip event j (extra spurious event), carry forward
            if j > 0 and dp[i, j - 1] < dp[i, j]:
                dp[i, j] = dp[i, j - 1]
                back[i, j] = back[i, j - 1]

    # Traceback
    assigned = np.full(n, -1, dtype=int)
    j = m
    for i in range(n, 0, -1):
        b = back[i, j]
        if b > 0:
            assigned[i - 1] = b - 1  # convert to 0-indexed event
            j = b - 1
        elif b == -1:
            pass  # slot skipped, j unchanged
        else:
            pass

    pairs = assigned >= 0
    times = np.full(n, np.nan)
    times[pairs] = events_sec[assigned[pairs]]

    residuals = np.abs(times[pairs] - ref_shifted[pairs])
    region_quality = _region_quality(pairs)

    metrics = {
        "method": "dp",
        "offset": offset,
        "scale": 1.0,
        "mismatch_penalty": mismatch_penalty,
        "total_cost": float(dp[n, m]),
        "match_rate": float(np.sum(pairs) / n),
        "mean_residual": float(np.mean(residuals)) if len(residuals) else np.nan,
        "std_residual": float(np.std(residuals)) if len(residuals) else np.nan,
        "n_matched": int(np.sum(pairs)),
        "n_reference": n,
        "n_events": m,
    }

    return pairs, times, region_quality, metrics


# ---------------------------------------------------------------------------
# Quality labeling
# ---------------------------------------------------------------------------

def _region_quality(
    pairs: np.ndarray,
    bad_threshold: int = 3,
) -> np.ndarray:
    """
    Label each reference slot 0 (good) or 1 (bad).

    A slot is labeled bad if it is unmatched AND within a run of
    >= bad_threshold consecutive unmatched slots. Isolated unmatched
    slots in an otherwise well-matched region remain labeled good.

    Parameters
    ----------
    pairs : bool array, shape (n,)
    bad_threshold : consecutive unmatched slots to declare a bad region

    Returns
    -------
    quality : int array, shape (n,), 0=good 1=bad
    """
    n = len(pairs)
    quality = np.zeros(n, dtype=int)

    i = 0
    while i < n:
        if not pairs[i]:
            # Count run of unmatched slots
            j = i
            while j < n and not pairs[j]:
                j += 1
            run_len = j - i
            if run_len >= bad_threshold:
                quality[i:j] = 1
            i = j
        else:
            i += 1

    return quality


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

    Each step is a callable with signature:
        step(rising, falling, sample_rate) -> (rising, falling)

    Steps are applied in order. Each receives the output of the previous.

    Parameters
    ----------
    rising : np.ndarray, shape (m,), sample indices of rising edges
    falling : np.ndarray, shape (m,), sample indices of falling edges
    sample_rate : float, samples per second
    steps : list of callable

    Returns
    -------
    rising : np.ndarray, filtered
    falling : np.ndarray, filtered
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
    """Keep events where the interval to the previous rising edge >= min_samples."""
    def _filter(rising, falling, sample_rate):
        if len(rising) < 2:
            return rising, falling
        intervals = np.diff(rising)
        # First event has no predecessor — keep it unconditionally
        mask = np.concatenate([[True], intervals >= min_samples])
        return rising[mask], falling[mask]
    return _filter


def filter_max_interval(max_samples: int):
    """Keep events where the interval to the previous rising edge <= max_samples."""
    def _filter(rising, falling, sample_rate):
        if len(rising) < 2:
            return rising, falling
        intervals = np.diff(rising)
        mask = np.concatenate([[True], intervals <= max_samples])
        return rising[mask], falling[mask]
    return _filter


# ---------------------------------------------------------------------------
# Public matcher
# ---------------------------------------------------------------------------

def match(
    reference: np.ndarray,
    rising: np.ndarray,
    sample_rate: float,
    tolerance: float,
    fallback: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Match a noisy detector event stream to a reference timing template.

    Parameters
    ----------
    reference : np.ndarray, shape (n,)
        Timestamps in seconds from session start (time.monotonic() - start).
    rising : np.ndarray, shape (m,)
        Sample indices of detector pulse rising edges.
    sample_rate : float
        Samples per second. Used to convert rising to seconds.
    tolerance : float
        Maximum allowed deviation in seconds for a match to be accepted.
    fallback : bool
        If True, attempt progressively more complex matchers if the
        previous tier fails to meet quality threshold (match_rate < 0.9).
        If False, raise ValueError if the primary matcher fails.

    Returns
    -------
    pairs : np.ndarray, shape (n,), bool
        True where a detector event was matched to the reference slot.
    times : np.ndarray, shape (n,), float
        Matched event times in seconds. nan where unmatched.
    region_quality : np.ndarray, shape (n,), int
        0 = good, 1 = bad (consecutive unmatched slots).
    metrics : dict
        Matching diagnostics. Always includes: method, match_rate,
        mean_residual, std_residual, n_matched, n_reference, n_events.
    """
    QUALITY_THRESHOLD = 0.9

    events_sec = rising / sample_rate

    # Tier 1: linear fit
    pairs, times, region_quality, metrics = _match_linear(
        reference, events_sec, tolerance
    )

    if metrics["match_rate"] >= QUALITY_THRESHOLD:
        return pairs, times, region_quality, metrics

    if not fallback:
        raise ValueError(
            f"Linear matcher achieved match_rate={metrics['match_rate']:.3f}, "
            f"below threshold {QUALITY_THRESHOLD}. "
            f"Re-run with fallback=True to attempt greedy and DP matchers."
        )

    # Tier 2: greedy, seeded with linear fit offset/scale
    pairs, times, region_quality, metrics = _match_greedy(
        reference, events_sec, tolerance,
        offset=metrics.get("offset"),
        scale=metrics.get("scale"),
    )

    if metrics["match_rate"] >= QUALITY_THRESHOLD:
        return pairs, times, region_quality, metrics

    # Tier 3: DP
    pairs, times, region_quality, metrics = _match_dp(
        reference, events_sec, tolerance
    )

    if metrics["match_rate"] >= QUALITY_THRESHOLD:
        return pairs, times, region_quality, metrics

    raise ValueError(
        f"All matchers failed. Best match_rate={metrics['match_rate']:.3f}. "
        f"Check tolerance, sample_rate, or data quality."
    )
