"""
wfmsynth.measure — attributes measured FROM the generated waveform, not computed from the
knobs. The two differ, and that difference is exactly where silent label noise comes from,
so ground-truth labels for training should be measured here rather than read off the recipe.

Eye height is offered under **two named definitions** because they disagree when it matters:

  - ``contour``  the measured vertical opening — the gap between the worst-case samples of
                 adjacent levels. Honest about deterministic ISI (a discrete echo makes
                 distinct branches, not a smear).
  - ``sigma``    a 3-sigma construction per level. Agrees with ``contour`` for roughly
                 Gaussian vertical distributions and **overstates** the opening when ISI is
                 deterministic. Report which one you mean.

All measurements fold the record at the symbol period implied by ``grid`` (fs/baud), search
the sampling phase, and cluster samples into levels — no knowledge of the transmitted bits.

A SWEPT-WAVELENGTH record (`wfmsynth.swept`) is read back by three more, which know nothing of
the recipe either: `swept_map` turns the reference interferometer's fringes into wavelength
against time, `transmission` puts a detector record on that axis, and `resonances` reads each
dip's wavelength, width, Q and extinction off it.
"""
from __future__ import annotations

import numpy as np


def voltage(x, who):
    """`x` as it is, unless it is COMPLEX: every measure in this module and in `wfmsynth.eye`
    folds, thresholds or clusters a real voltage, and handed an I/Q record it would cast it
    to float -- drop Q under a numpy warning -- and return a plausible number for the I rail
    alone. So a complex record is refused by name, with what measures one."""
    if np.iscomplexobj(x):
        raise ValueError(
            f"{who}: the record is complex baseband (I/Q), and {who} measures a real voltage "
            f"-- it would silently keep the I rail and drop Q. An I/Q record is measured by "
            f"wfmsynth.rf: rf.evm (error against the symbols), rf.constellation_density, "
            f"rf.spectrum, rf.ccdf. To measure the modulated carrier as a voltage, "
            f"upconvert(fc_hz=...) first; to measure one rail on purpose, pass x.real or "
            f"x.imag. (If this complex record is an OPTICAL FIELD, none of that applies: "
            f"photodetect() turns it into the real photocurrent a receiver measures.)")
    return x


def sample_at_phase(x, spb, phase):
    """Sample one point per symbol at ``round(k*spb + phase)``. ``spb`` may be non-integer."""
    voltage(x, "sample_at_phase")
    n_sym = int((len(x) - 1 - phase) / spb)
    idx = np.round(np.arange(n_sym) * spb + phase).astype(int)
    idx = idx[(idx >= 0) & (idx < len(x))]
    return x[idx]


def _nearest_level(samps, centers):
    """Index of the nearest of ``centers`` for every sample, lowest index winning a tie —
    exactly ``|samps[:,None]-centers|.argmin(1)``, without building the ``(n, k)`` array
    that dominated a deep record's eye measurement.

    A running minimum over the ``k`` centers instead of one ``k``-wide reduction. Ties keep
    the lower index because the update is strictly ``<``, which is argmin's own rule, so
    this is equal by construction rather than to within a tolerance.
    """
    k = len(centers)
    if k < 2:
        return np.zeros(len(samps), dtype=np.intp)
    lab = np.zeros(len(samps), dtype=np.intp)
    best = np.abs(samps - centers[0])
    d = np.empty_like(best)
    for j in range(1, k):
        np.abs(np.subtract(samps, centers[j], out=d), out=d)
        closer = d < best
        lab[closer] = j
        np.minimum(best, d, out=best)
    return lab


def _cluster_levels(samps, k):
    """Assign samples to ``k`` amplitude levels (quantile init + a few Lloyd iterations).
    Returns sorted centers and per-sample labels."""
    centers = np.quantile(samps, [(i + 0.5) / k for i in range(k)])
    for _ in range(8):
        lab = _nearest_level(samps, centers)
        new = np.array([samps[lab == j].mean() if np.any(lab == j) else centers[j]
                        for j in range(k)])
        new.sort()
        if np.allclose(new, centers):
            break
        centers = new
    lab = _nearest_level(samps, centers)
    return centers, lab


def _eye_at_phase(x, spb, phase, levels, defn):
    s = sample_at_phase(x, spb, phase)
    if len(s) < levels * 4:
        return -np.inf
    _, lab = _cluster_levels(s, levels)
    gaps = []
    for j in range(levels - 1):
        lo, hi = s[lab == j], s[lab == j + 1]
        if len(lo) == 0 or len(hi) == 0:
            return -np.inf
        if defn == "contour":
            gaps.append(hi.min() - lo.max())
        elif defn == "sigma":
            gaps.append((hi.mean() - 3 * hi.std()) - (lo.mean() + 3 * lo.std()))
        else:
            raise ValueError(f"unknown eye definition {defn!r} (use 'contour' or 'sigma')")
    return min(gaps)


def eye_height(x, grid, levels, defn="contour", n_phases=32):
    """Measured eye height at the best sampling phase. ``defn`` is 'contour' (measured
    opening) or 'sigma' (3-sigma construction) — see the module docstring; they diverge
    under deterministic ISI.

    ``levels`` is REQUIRED and has no default: 2 for NRZ, 4 for PAM4, N for PAM-N. It is a
    property of the record rather than a preference, and the caller always knows it because the
    caller chose the carrier. A default would be silently wrong for half of all records and would
    look plausible either way -- MEASURED on one NRZ record, reading it as four levels gave 0.0017
    against the 0.0432 it actually has, a 25x understatement with nothing to say so.
    """
    voltage(x, "eye_height")
    spb = grid.samples_per_ui
    return max(_eye_at_phase(x, spb, ph, levels, defn)
               for ph in np.linspace(0, spb, n_phases, endpoint=False))


def best_phase(x, grid, levels, defn="contour", n_phases=32):
    """The sampling phase (in samples, within one UI) that maximizes the eye opening.

    ``levels`` is required, for the reason `eye_height` gives."""
    voltage(x, "best_phase")
    spb = grid.samples_per_ui
    phases = np.linspace(0, spb, n_phases, endpoint=False)
    return float(phases[int(np.argmax([_eye_at_phase(x, spb, ph, levels, defn)
                                       for ph in phases]))])


def pattern_period(symbols, max_lag=None):
    """Detect the repetition period of a symbol sequence from its (overlap-normalized)
    autocorrelation — the lag of the strongest correlation peak beyond lag 0. A carrier
    claiming a standard pattern must lock to a single sharp peak at its declared period;
    that peak is what an analyser pattern-locks onto. Returns ``(period, peak, ac)`` where
    ``ac`` is the normalized autocorrelation (ac[0] == 1). FFT-based, O(n log n)."""
    if np.iscomplexobj(symbols):
        raise ValueError(
            "pattern_period: the symbols are complex (an I/Q constellation), and the period "
            "is found from a real autocorrelation, which would keep I and drop Q. Pass the "
            "bits, or one rail (symbols.real), whichever repeats.")
    s = np.asarray(symbols, float)
    s = s - s.mean()
    n = len(s)
    max_lag = int(max_lag) if max_lag else n // 2
    v = np.dot(s, s) / n + 1e-12
    f = np.fft.rfft(s, 2 * n)
    raw = np.fft.irfft(f * np.conj(f), 2 * n)[:max_lag + 1].real     # linear autocorr, lags 0..max_lag
    ac = raw / ((n - np.arange(max_lag + 1)) * v)                    # overlap-normalized
    ac[0] = 1.0
    lag = int(np.argmax(ac[1:])) + 1
    return lag, float(ac[lag]), ac


def align_symbols(x, grid, tx, levels=4, defn="contour", n_phases=32):
    """Realized integer-symbol alignment between the transmitted stream ``tx`` and the
    sampled output. A minimum-phase (causal) channel has group delay, so the sampled
    output symbol *k* corresponds to transmitted symbol *k - offset*; any per-symbol
    statistic needs this realignment or it silently compares the wrong pairs.

    Returns ``(offset, corr_aligned, corr_zero)``: the best integer offset, the normalized
    tx/output correlation at that offset, and the correlation at zero offset (skipping the
    realignment) — so the cost of ignoring the delay is visible.

    The offset is chosen by ``argmax(|corr|)``, not ``argmax(corr)``: on an INVERTED record (a
    differential pair wired backwards, or any inverting stage) every true-alignment correlation
    is negative, and a plain argmax locks onto noise instead. The magnitude identifies the
    alignment and the returned sign identifies the polarity — the only way to detect a swapped
    pair, since eye height, peak-to-peak and RMS are all identical under negation."""
    voltage(x, "align_symbols")
    spb = grid.samples_per_ui
    y = sample_at_phase(x, spb, best_phase(x, grid, levels, defn, n_phases))
    m = min(len(y), len(tx))
    yc = y[:m] - y[:m].mean()
    sc = np.asarray(tx[:m], float) - np.asarray(tx[:m], float).mean()
    denom = np.sqrt((yc ** 2).sum() * (sc ** 2).sum()) + 1e-12
    corr = np.correlate(yc, sc, mode="full") / denom
    lags = np.arange(-m + 1, m)
    k = int(np.argmax(np.abs(corr)))
    corr_zero = float(corr[lags.tolist().index(0)])
    return int(lags[k]), float(corr[k]), corr_zero


def ground_truth(x, grid, tx=None, levels=4):
    """Consolidated ground-truth labels MEASURED from ``x`` (never computed from the knobs):
    eye height under both named definitions, the realized sampling phase, and — when the
    transmitted stream ``tx`` is supplied — the realized integer-symbol alignment offset.
    ``Signal.ground_truth()`` fills ``tx`` in automatically from the recipe."""
    voltage(x, "ground_truth")
    gt = attributes(x, grid, levels=levels)
    gt["best_phase"] = best_phase(x, grid, levels=levels)
    if tx is not None:
        off, ca, cz = align_symbols(x, grid, tx, levels=levels)
        gt["align_offset"] = off
        gt["align_corr"] = ca
        gt["align_corr_at_zero"] = cz
    return gt


def attributes(x, grid, levels=4):
    """A dict of realized attributes measured from ``x`` — both eye definitions plus basic
    amplitude stats. Use as the ``metrics`` source for realized-vs-requested labelling."""
    voltage(x, "attributes")
    return {
        "eye_contour": eye_height(x, grid, levels=levels, defn="contour"),
        "eye_sigma": eye_height(x, grid, levels=levels, defn="sigma"),
        "ptp": float(np.ptp(x)),
        "rms": float(np.sqrt(np.mean(x ** 2))),
    }


# ------------------------------------------------------------- a swept-wavelength record
def _fringe_end(c, guess, x, rising):
    """The phase of ``c = cos(phase)`` over the part fringe at one end of a record, WALKING
    AWAY from the whole fringes it joins: `c`, the rough `guess` of the phase and `x`, the
    distance walked (zero at the join), all in that order, with the phase taken as growing
    along the walk. `rising` says whether the walk sets out up a fringe (after the last
    counted rise) or down one (backwards from the first).

    Each sample's arccosine allows two phases a turn. Which one is read from the fringe's own
    SHAPE and not from the guess, which a sweep whose rate changes here (a ripple, a bow)
    puts out by more than the quarter turn that would decide it: a part fringe is under one
    turn, so it climbs to at most one crest and falls to at most one trough, in an order the
    end it is at fixes, and the phase is -a before a crest, +a after it, and a turn on past
    a trough. The crest is sought in the stretch that stands above half height and the
    trough in the stretch below minus half, as the whole fringes are counted.

    A turn is the extreme of the samples AVERAGED over an eighth of a turn, not the extreme
    sample: with detector noise on a walk still climbing when the record ends, some sample
    before the last is the highest, and all after it would be read as coming down from a
    crest never reached (a quarter of a fringe out, at 1 % noise). Where the averages are
    still climbing at the last whole one, the walk has either not turned or turned in the
    few samples after; both are read, and the one whose cosine lies closer to the samples
    is kept.

    What a reading says the guess is out by is fitted as a smooth curve in `x` (zero at the
    join), weighted by the fringe's slope, and the curve is then moved to where its cosine
    is closest to the samples, so no one noisy sample near a crest decides the answer.
    Under eight samples there is no fit, and the reading itself is returned."""
    n = len(c)
    if not n:
        return guess
    raw, c = np.asarray(c, float), np.clip(c, -1.0, 1.0)
    a = np.arccos(c)
    side = np.sign(c) * (np.abs(c) > 0.5)
    held = side[np.maximum.accumulate(np.where(side != 0, np.arange(n), 0))]
    # until it first stands below minus half, the walk is on the stretch it set out on:
    # up to the crest, or down to that trough
    below = np.flatnonzero(side < 0)
    held[:below[0] if len(below) else n] = 1.0 if rising else -1.0
    # `c` averaged over an eighth of a turn at the guess's rate, each average whole
    h = int(np.pi / 8 * (n - 1) / max(abs(guess[-1] - guess[0]), 1e-300)) if n > 1 else 0
    mean = np.convolve(c, np.full(2 * h + 1, 1.0 / (2 * h + 1)), "valid") if n > 2 * h else c

    def turning(stretch, up):
        """Where the walk turns in this stretch, and where else it may have."""
        whole = stretch[(stretch >= h) & (stretch < n - h)]
        if len(whole):
            level = mean[whole - h]
            k = int(whole[np.argmax(level if up else -level)])
            if k < n - 1 - h:
                return k, None
            # on the last whole average the walk is still on its way to the turn, or has
            # made it among the samples after: the extreme of those is where, if it has
            rest = stretch[stretch >= k]
            return None, int(rest[np.argmax(c[rest] if up else -c[rest])]) if h else None
        if not len(stretch) or stretch[-1] == n - 1:
            return None, None                 # too short a walk to have turned
        return int(stretch[np.argmax(c[stretch] if up else -c[stretch])]), None

    (crest, or_crest), (trough, or_trough) = (turning(np.flatnonzero(held > 0), True),
                                              turning(np.flatnonzero(held < 0), False))
    i, x = np.arange(n), np.asarray(x, float)
    basis = np.stack([x, x * x], axis=1)

    def reading(crest, trough):
        """The phases these turns give, and how far their cosine lies from the samples."""
        if rising:                       # ... crest, then trough
            turn = np.zeros(n) if trough is None else (i > trough).astype(float)
            sign = np.where((i <= (n if crest is None else crest)) | (turn > 0), -1.0, 1.0)
        else:                            # ... trough, then crest
            after = np.zeros(n, bool) if crest is None else i > crest
            turn = (np.zeros(n) if trough is None else (i > trough).astype(float))
            sign = np.where((i <= (n if trough is None else trough)) | after, 1.0, -1.0)
        read = sign * a + 2 * np.pi * turn
        read += 2 * np.pi * np.round((guess[0] - read[0]) / (2 * np.pi))
        if n < 8:
            return 0.0, read
        w = np.sqrt(1.0 - c * c) + 1e-3
        coef = np.linalg.lstsq(basis * w[:, None], (read - guess) * w, rcond=None)[0]

        def fitted(coef):                 # not `basis @ coef`: numpy 2.0 on macOS warns
            return guess + coef[0] * x + coef[1] * x * x

        # ... then moved to where its cosine lies closest to the samples themselves, which
        # asks no sample which side of a turn it is on
        miss = np.sum((raw - np.cos(fitted(coef))) ** 2)
        for _ in range(8):
            phase = fitted(coef)
            step = np.linalg.lstsq(basis * np.sin(phase)[:, None], np.cos(phase) - raw,
                                   rcond=None)[0]
            closer = np.sum((raw - np.cos(fitted(coef + step))) ** 2)
            if not closer < miss:
                break
            coef, miss = coef + step, closer
        return miss, fitted(coef)

    best = reading(crest, trough)
    if or_crest is not None or or_trough is not None:
        other = reading(crest if or_crest is None else or_crest,
                        trough if or_trough is None else or_trough)
        best = other if other[0] < best[0] else best
    return best[1]


def swept_map(ref, grid, delay_s, at_s, wavelength_nm, rising=True):
    """The wavelength at every sample of a sweep, RECOVERED from its reference channel: the
    photocurrent `ref` of an interferometer whose arms differ by `delay_s`
    (`wfmsynth.recipes.mzi`), on the record's `grid` (its sample rate is what is read,
    as by every measure here).

    The fringe is cos(2*pi*nu*delay_s), so its phase -- taken from the analytic signal of
    the fringe about its mid level, and unwrapped -- counts optical frequency in units of
    1/delay_s. A count is relative: `at_s` and `wavelength_nm` say what the wavelength was
    at one time (the sweep's stated start, or the instant a gas line was crossed). `rising`
    says which way the wavelength went; a fringe cannot tell.

    What it cannot see, it gets wrong, as a real k-clock does: a mode hop moves the
    wavelength without the fringe count following, so everything after it is off by the hop
    less a whole number of fringes; and a sweep that reverses is read as carrying on.

    The analytic signal is taken over a WHOLE number of fringes -- from the first upward
    crossing of the mid level to the last, where the record joins onto itself -- because
    over a cut fringe it is wrong by a large part of a fringe at both ends. The part
    fringe left at each end is read from the fringe's own shape, its crest and its trough
    (`_fringe_end`), so a sweep whose rate is changing there is still followed: measured
    0.02 of a fringe at worst with a ripple that moves the rate by 30 % inside the last
    fringe; and with detector noise of 1 % and 2 % of the fringe's height on a rippled and
    bowed sweep, 0.023 and 0.039, no worse than the whole fringes beside it
    (tests/test_swept.py). An anchor inside the first part fringe (`at_s=0`) rests on that
    reading, and so does every sample."""
    from scipy.signal import hilbert
    ref = np.asarray(voltage(ref, "swept_map"), float)
    n, fs = len(ref), grid.fs
    if not (float(delay_s) > 0 and float(wavelength_nm) > 0 and float(fs) > 0):
        raise ValueError("swept_map: the grid's fs, delay_s (the interferometer's arm "
                         "difference, in seconds) and wavelength_nm (the anchor) are positive.")
    k = float(at_s) * float(fs)
    if not 0 <= k <= n - 1:
        raise ValueError(f"swept_map: at_s={at_s!r} is the time of the anchor and lies "
                         f"inside the record (0 to {(n - 1) / fs:.6g} s).")
    amp = 0.5 * (ref.max() - ref.min())
    c = (ref - 0.5 * (ref.max() + ref.min())) / (amp or 1.0)             # cos(phase)
    # where a fringe RISES through half its height, having been below minus half: a rise
    # that noise at a crossing cannot fake
    side = np.sign(c) * (np.abs(c) > 0.5)
    last = np.maximum.accumulate(np.where(side != 0, np.arange(n), 0))
    side = side[last]
    up = np.flatnonzero((side[:-1] < 0) & (side[1:] > 0)) + 1
    i0, i1 = (int(up[0]), int(up[-1])) if len(up) else (0, 0)
    phase = np.empty(n)
    if i1 > i0:
        # over whole fringes the mean IS the mid level and the rms the height, which the
        # record's extremes are not once there is noise on it
        c = (ref - ref[i0:i1].mean()) / (np.sqrt(2.0) * ref[i0:i1].std() or 1.0)
        phase[i0:i1] = np.unwrap(np.angle(hilbert(c[i0:i1])))
    if i1 - i0 < 2 or phase[i1 - 1] - phase[i0] < 3 * np.pi:
        raise ValueError("swept_map: the reference shows under two whole fringes, so there "
                         "is nothing to count. Is this the interferometer's channel, and "
                         "does the sweep cover several times 1/delay_s of optical frequency?")
    # the part fringe before the first rise and after the last: guessed from the rate of
    # the whole fringe beside it, then read off the fringe itself
    j0 = i0 + int(np.argmax(phase[i0:i1] >= phase[i0] + 2 * np.pi))
    j1 = i1 - 1 - int(np.argmax(phase[i0:i1][::-1] <= phase[i1 - 1] - 2 * np.pi))
    before, after = np.arange(i0)[::-1] - i0, np.arange(i1, n) - (i1 - 1)
    # the head is walked BACKWARDS from the first rise, where minus the phase is what rises
    phase[:i0] = -_fringe_end(
        c[:i0][::-1], -(phase[i0] + (phase[j0] - phase[i0]) / (j0 - i0) * before),
        -before, rising=False)[::-1]
    phase[i1:] = _fringe_end(
        c[i1:], phase[i1 - 1] + (phase[i1 - 1] - phase[j1]) / (i1 - 1 - j1) * after, after,
        rising=True)
    turns = (phase - np.interp(k, np.arange(n), phase)) / (2 * np.pi)
    c_nm = 299792458.0e9
    return c_nm / (c_nm / float(wavelength_nm) + (-1.0 if rising else 1.0) * turns / delay_s)


def transmission(x, wavelength_nm, power=1.0):
    """A swept record against WAVELENGTH: ``(wavelength_nm, t)``, both ascending in
    wavelength, with ``t = x / power``.

    `x` is the detector's record and `wavelength_nm` the wavelength at each of its samples:
    the truth (`ws.swept.wavelength_map(sig)["wavelength_nm"]`) or what the reference
    channel recovered (`swept_map`). `power` is what the detector reads with NO DEVICE in
    the path -- a number, or a record of a tap on the laser, sample for sample -- so it
    holds the laser's power, the detector's responsivity and every loss that is not the
    device's. With it right, `t` is 1 away from a resonance and `resonances(level=1.0)`
    is the honest call; left at 1.0 on a bench with any of those, `t` is in the detector's
    units and `resonances` is called WITHOUT `level` (it then takes the spectrum's own)."""
    x = np.asarray(voltage(x, "transmission"), float)
    lam = np.asarray(wavelength_nm, float)
    if lam.shape != x.shape:
        raise ValueError(f"transmission: {x.shape[0]} samples and {lam.size} wavelengths. "
                         f"The map has one wavelength per sample of the record; after an "
                         f"`acquire` resample the map onto the stored grid first.")
    order = np.argsort(lam, kind="stable")
    return lam[order], (x / power)[order]


def resonances(wavelength_nm, t, depth=0.1, level=None):
    """The dips of a transmission spectrum, read off the curve: a list of
    ``{"wavelength_nm", "fwhm_nm", "q", "extinction_db", "t_min"}`` in order of wavelength.

    A dip is a minimum that stands out by at least `depth`. Its bottom is the vertex of a
    parabola through the three lowest samples; its width the distance between the two
    crossings (interpolated) of half way from the bottom up to `level`, the transmission
    away from resonance; ``q = wavelength / width`` and ``extinction_db`` is `level` over
    the bottom. `level` is the OFF-RESONANCE BASELINE of `t`, in `t`'s own units: 1.0 only
    for a spectrum already divided by what the detector reads without the device
    (`transmission(power=)`). Left out it is the spectrum's median, which a fringe does not
    move but a dip's own skirts pull down: on a span of 30 widths the width reads 0.4 %
    narrow, so state `level` when it is known. A stated `level` the spectrum never comes
    half way up to from a dip is refused: it is the baseline of some other record (an
    insertion loss not divided out), and the width would have no crossing to be read from.
    A SMALLER loss left in the record cannot be told from a ring of low finesse, whose
    spectrum never comes back up to its baseline either (0.94 at a finesse of 10), so it
    is read, and read wrong: 1 dB left in with `level=1.0` gives the width 30 % wide and
    the extinction 1 dB too deep. `level` is a statement about the record; it is not checked.
    The free spectral range is the spacing of successive `wavelength_nm`. It needs several
    samples across a dip, and a dip clear of its neighbours at half depth."""
    from scipy.signal import find_peaks
    lam, t = np.asarray(wavelength_nm, float), np.asarray(t, float)
    base = float(np.median(t)) if level is None else float(level)
    out = []
    for i in find_peaks(-t, prominence=depth)[0]:
        a, b, c = t[i - 1], t[i], t[i + 1]
        dx = 0.5 * (a - c) / (a - 2 * b + c)                   # vertex, in samples from i
        t_min = b - 0.25 * (a - c) * dx
        centre = lam[i] + dx * (lam[i + 1] - lam[i] if dx >= 0 else lam[i] - lam[i - 1])
        half, edges = 0.5 * (base + t_min), []
        if level is not None and not t.max() >= half:
            raise ValueError(
                f"resonances: level={level!r} is the transmission away from resonance, and "
                f"this spectrum never rises above {t.max():.4g}: half way up from the dip at "
                f"{centre:.6g} nm ({half:.4g}) is not on it. The record carries a loss or a "
                f"scale that is not the device's (insertion loss, responsivity, laser "
                f"power). Divide it out -- transmission(x, wavelength_nm, power=<what the "
                f"detector reads with no device>) -- or leave level= out, and the "
                f"spectrum's own median is the baseline.")
        for step in (-1, 1):
            j = i
            while 0 < j < len(t) - 1 and t[j] < half:
                j += step
            lo = j - step                                      # t[lo] < half <= t[j]
            edges.append(lam[lo] + (half - t[lo]) / (t[j] - t[lo]) * (lam[j] - lam[lo])
                         if t[j] >= half else np.nan)
        width = edges[1] - edges[0]
        out.append({"wavelength_nm": float(centre), "fwhm_nm": float(width),
                    "q": float(centre / width),
                    "extinction_db": float(10 * np.log10(base / max(t_min, 1e-300))),
                    "t_min": float(t_min)})
    return out
