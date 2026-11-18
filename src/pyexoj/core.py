"""pyexoj — a Python port of ExoJ 1.9.2 (exocytosis event detection in fluorescence time series).

ExoJ: Liu J., Verweij F.J., van Niel G., Galli T., Danglot L., Bun P., "ExoJ: an ImageJ2/Fiji plugin
for automated spatiotemporal detection and analysis of exocytosis", J. Cell Sci. 137(20):jcs261938
(2024). Copyright (C) 2022 LIU Junjun, BUN Philippe. ExoJ 1.9.2 (Zenodo 22043601) is GPL-3.0-or-later.

This port was written from the decompiled ExoJ_1.9.2.jar (Vineflower 1.12.0; no 1.9.2 source is
published), from ImageJ 1.54p ij.measure.CurveFitter (fit type 11) and Apache Commons Math 3.6.1
SimpleRegression / LevenbergMarquardtOptimizer. It is a derived work of ExoJ and is distributed under
the GNU General Public License, version 3 or (at your option) any later version.

    This program is free software: you can redistribute it and/or modify it under the terms of the
    GNU General Public License as published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version. This program is distributed in the hope that it
    will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See <https://www.gnu.org/licenses/>.

Exactness: steps 1 and 2 reproduce ExoJ's float32/float64 arithmetic in the same order (same spots,
same tracks). Step 3's decay fit solves the same 1-D least-squares problem as ImageJ's CurveFitter
(a, c by linear regression, b searched) but with Brent instead of ImageJ's randomised simplex, and the
optional 2-D Gaussian uses MINPACK LM (scipy.leastsq) like Commons Math; fitted values agree to the
original's own run-to-run scatter, not bit for bit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from decimal import Decimal, ROUND_HALF_EVEN

import numpy as np
from scipy import ndimage, optimize

F32 = np.float32
_W2, _W1, _W0 = F32(0.0625), F32(0.25), F32(0.375)
_AVAILABLE_SCALE = (1, 3, 7, 13, 25)


def jround_f(x: float) -> int:
    """Java Math.round(float): floor(x + 0.5f) in float arithmetic."""
    return int(math.floor(float(F32(x) + F32(0.5))))


def jround_d(x):
    """Java Math.round(double) on an array: floor(x + 0.5)."""
    return np.floor(x + 0.5)


def jmedian(a: np.ndarray):
    """ExoJ FindMedian / FindMedianD: the (n//2)-th smallest element (upper median)."""
    n = a.size
    return np.partition(a.ravel(), n // 2)[n // 2]


# ----------------------------------------------------------------------------------- parameters

@dataclass
class Params:
    """Field names follow ExoJ's Parameters; defaults are ExoJ 1.9.2's."""
    pixelSize: float = 1.0
    timeInterval: float = 1.0
    # step 1
    minSize: int = 4
    maxSize: int = 8
    SNR_Vesicle: float = 3.0
    WaveletFilter: bool = True
    bleach_correct: bool = False
    deltamovie: bool = False
    # step 2
    SpatialsearchRadius: int = 2
    TemporalSearchDepth: int = 1
    minimalEventSize: int = 2
    # step 3
    useMinimalPointsForFitter: bool = True
    minimalPointsForFitter: int = 5
    useExpandFrames: bool = True
    expand_frames_L: int = 4
    expand_frames_R: int = 4
    fixedExpand: bool = True
    useMinSNR: bool = True
    min_SNR: float = 3.0
    useMaxTauFrames: bool = False
    max_tau: float = 20.0
    useMinTauFrames: bool = False
    min_tau: float = 1.0
    useMinDecayFitterR2: bool = True
    min_decay_fitter_r2: float = 0.8
    useMaxSize: bool = False
    max_estimated_size: float = 8.0
    useMinSize: bool = False
    min_estimated_size: float = 4.0
    useGaussienfitter2DR2: bool = False
    min_gaussienfitter2DR2: float = 0.75
    useMaxDisplacement: bool = False
    MaxDisplacement: float = 3.0

    def scale_list(self) -> list[bool]:
        """Parameters.setScaleList(minSize, maxSize) (as the size fields' listeners call it)."""
        mn = min(max(self.minSize, 1), 25)
        mx = min(max(self.maxSize, 1), 25)
        sl = [mn - 1 <= s <= mx - 1 for s in _AVAILABLE_SCALE]
        if not any(sl):
            for i, s in enumerate(_AVAILABLE_SCALE):
                if s >= mn - 1:
                    sl[max(0, i - 1)] = True
                    break
        return sl

    @property
    def needs_gaussian(self) -> bool:
        return self.useGaussienfitter2DR2 or self.useMaxSize or self.useMinSize


# ----------------------------------------------------------------------------------- step 1

def bleaching_correction(stack: np.ndarray) -> np.ndarray:
    """PhotolechingCorrection.MeshBleachingCorrection (1.9.2): pixel-wise weighted detrend."""
    T = stack.shape[0]
    if T < 2:
        return stack.copy()
    f = stack.astype(np.float32)
    n = f[0].size
    # Java: double sum += float pixel, sequential
    mean = np.array([np.cumsum(fr.ravel().astype(np.float64))[-1] / n for fr in f])
    mn, mx = mean[0], mean[0]
    for v in mean:
        mn = min(mn, v) if v < mn else mn
        mx = v if v > mx else mx
    mean_global = 0.0
    for v in mean:
        mean_global += v
    mean_global /= T
    rng = mx - mn
    rel = abs(rng) / max(abs(mean_global), 1.0)
    if not (math.isfinite(rel) and not rel <= 1e-7):
        return stack.copy()
    weight = (f[0] - f[-1]).astype(np.float64)       # float subtraction, then promoted
    out = np.empty_like(stack)
    for t in range(T):
        corr = (mean[t] - mean_global) * weight / rng
        v = f[t].astype(np.float64) - corr
        v[~np.isfinite(v) | (v < 0.0)] = 0.0
        if stack.dtype == np.uint8:
            out[t] = (jround_d(np.minimum(255.0, v)).astype(np.int64) & 0xFF).astype(np.uint8)
        elif stack.dtype == np.uint16:
            out[t] = (jround_d(np.minimum(65535.0, v)).astype(np.int64) & 0xFFFF).astype(np.uint16)
        else:
            out[t] = v.astype(np.float32)
    return out


def abs_delta_movie(stack: np.ndarray) -> np.ndarray:
    """DeltaMovie.getAbsDeltaMovie: |F(t) - F(t-1)|, first frame 0 (float32)."""
    f = stack.astype(np.float32)
    d = np.zeros_like(f)
    d[1:] = np.abs(f[1:] - f[:-1])
    return d


def _reflect_index(n: int, off: int) -> np.ndarray:
    i = np.arange(n) + off
    i = np.where(i < 0, -i - 1, i)
    return np.where(i >= n, 2 * n - i - 1, i)


def _smooth_axis(a: np.ndarray, step: int, axis: int) -> np.ndarray:
    """One pass of MicroWaveletFilter.filterAndSwap2D (B3 spline, holes = step, mirror borders)."""
    n = a.shape[axis]
    m2, m1 = _reflect_index(n, -2 * step), _reflect_index(n, -step)
    p1, p2 = _reflect_index(n, step), _reflect_index(n, 2 * step)
    t = lambda idx: np.take(a, idx, axis=axis)
    return _W2 * (t(m2) + t(p2)) + _W1 * (t(m1) + t(p1)) + _W0 * a


def _smooth_axis_fast(a: np.ndarray, step: int, axis: int) -> np.ndarray:
    """Same values and operation order as _smooth_axis, with a mirrored pad and views instead of np.take
    ('symmetric' padding = ExoJ's single reflection while 2*step <= n, which filter()'s size check ensures)."""
    n = a.shape[axis]
    pw = [(0, 0)] * a.ndim
    pw[axis] = (2 * step, 2 * step)
    ap = np.pad(a, pw, mode="symmetric")
    sl = lambda off: ap[(slice(None),) * axis + (slice(2 * step + off, 2 * step + off + n),)]
    return _W2 * (sl(-2 * step) + sl(2 * step)) + _W1 * (sl(-step) + sl(step)) + _W0 * a


def _smooth_axis_inplace(a: np.ndarray, step: int, axis: int) -> np.ndarray:
    """Same values and operation order as _smooth_axis; interior from slices into preallocated buffers
    (less memory traffic, which limits multi-process runs), the 2*step border lines via _smooth_axis's indices."""
    n = a.shape[axis]
    s = step
    out = np.empty_like(a)
    x = np.moveaxis(a, axis, -1)
    o = np.moveaxis(out, axis, -1)
    if n > 4 * s:
        I = slice(2 * s, n - 2 * s)
        oi = o[..., I]
        np.add(x[..., 0:n - 4 * s], x[..., 4 * s:n], out=oi)
        np.multiply(oi, _W2, out=oi)
        tmp = np.add(x[..., s:n - 3 * s], x[..., 3 * s:n - s])
        np.multiply(tmp, _W1, out=tmp)
        np.add(oi, tmp, out=oi)
        np.multiply(x[..., I], _W0, out=tmp)
        np.add(oi, tmp, out=oi)
        edge = np.concatenate([np.arange(0, 2 * s), np.arange(n - 2 * s, n)])
    else:
        edge = np.arange(n)
    full = lambda off: _reflect_index(n, off)[edge]
    t = lambda idx: np.take(x, idx, axis=-1)
    o[..., edge] = _W2 * (t(full(-2 * s)) + t(full(2 * s))) + _W1 * (t(full(-s)) + t(full(s))) + _W0 * t(edge)
    return out


def wavelet_filter(frame: np.ndarray, scale_list: list[bool], k: float, fast: bool = False) -> np.ndarray:
    """MicroWaveletFilter.filter: thresholded sum of the selected à-trous detail scales.
    fast=True: _smooth_axis_fast (opt-in; checked bit-identical, see README)."""
    smooth = {False: _smooth_axis, True: _smooth_axis_fast, "inplace": _smooth_axis_inplace}[fast]
    H, W = frame.shape
    max_scale = 0
    for i, s in enumerate(scale_list):
        if s:
            max_scale = i + 1
    out = np.zeros((H, W), np.float32)
    if max_scale == 0:
        return out
    min_size = 5 + (int(2.0 ** (max_scale - 1)) - 1) * 4
    if W < min_size or H < min_size:
        return out
    a_prev = frame.astype(np.float32)
    coeffs = []
    for s in range(1, max_scale + 1):
        step = int(2.0 ** (s - 1))
        a = smooth(smooth(a_prev, step, axis=1), step, axis=0)
        coeffs.append(a_prev - a)
        a_prev = a
    npix = W * H
    for depth in range(max_scale):
        d = coeffs[depth]
        if not scale_list[depth]:
            coeffs[depth] = np.zeros_like(d)
            continue
        q = npix // (1 << (2 * (depth + 1)))
        lam = math.sqrt(2.0 * math.log(q)) if q > 0 else float("nan")
        med = jmedian(d)
        mad = F32(jmedian(np.abs(d - med)) / F32(0.6745))
        thr = lam * float(mad) * k
        d = d.copy()
        d[~(d.astype(np.float64) >= thr)] = F32(0.0)
        coeffs[depth] = d
    v = np.zeros((H, W), np.float32)
    for j in range(max_scale):
        if scale_list[j]:
            v = v + coeffs[j]
    return v


def _disk(r: int) -> np.ndarray:
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def normalized_image(frame: np.ndarray) -> np.ndarray:
    """LocalMaximaDetector.getNormalizedImage(ImageProcessor) (FloatProcessor add/multiply in float)."""
    f = frame.astype(np.float32)
    mx, mn = f.ravel()[0], f.ravel()[0]
    mx, mn = max(mx, f.max()), min(mn, f.min())
    if mx == mn:
        return np.zeros_like(f)
    f = f + F32(-float(mn))
    return f * F32(1.0 / float(F32(mx - mn)))


def local_maxima_threshold(img: np.ndarray, k: float) -> np.float32:
    """LocalMaximaDetector.CalculateThreshold: median + k * MAD (float)."""
    med = jmedian(img)
    mad = float(jmedian(np.abs(img - med))) / 0.6745
    return F32(float(med) + k * mad)


@dataclass
class Spot:
    x: int
    y: int
    slice: int            # 1-based frame
    radius: int
    property: str = "Automatic"
    is_valid: bool = True
    max_den: int = 0
    min_den: int = 0
    crop: np.ndarray | None = None
    gfit: dict | None = None   # 2-D Gaussian fit cache


def _detect_frame(frame: np.ndarray, p: Params, roi, fast: bool):
    """Step-1 maxima of one frame: (ys, xs) in row-major order."""
    H, W = frame.shape
    spot_r = jround_f(p.minSize / 2.0)
    max_r = jround_f(p.maxSize / 2.0)
    y0, y1, x0, x1 = 2 * max_r, H - 2 * max_r, 2 * max_r, W - 2 * max_r
    if p.WaveletFilter:
        img = wavelet_filter(frame, p.scale_list(), p.SNR_Vesicle, fast=fast)
        thr = F32(1.0)
    else:
        img = normalized_image(frame)
        thr = local_maxima_threshold(img, p.SNR_Vesicle)
    mxf = ndimage.maximum_filter(img, footprint=_disk(spot_r), mode="constant", cval=-np.inf)
    cand = (img >= mxf) & ~(img < thr)
    sub = np.zeros_like(cand)
    if y1 > y0 and x1 > x0:
        sub[y0:y1, x0:x1] = cand[y0:y1, x0:x1]
    if roi is not None:
        sub &= roi
    return np.nonzero(sub)                  # row-major, like y-outer x-inner


_POOL_ARGS = None


def _detect_worker(t):
    stack, p, roi, fast = _POOL_ARGS
    return _detect_frame(stack[t], p, roi, fast)


def detect_spots(det_stack: np.ndarray, p: Params, roi: np.ndarray | None = None,
                 workers: int = 1, fast: bool = False, device: str | None = None) -> list[list[Spot]]:
    """Step 1 (VesicleDetectTask + LocalMaximaDetector.FindLocalMaxima), per frame.
    workers > 1: frames in parallel processes (fork; opt-in, checked identical). fast: see wavelet_filter.
    device: e.g. "cuda:0" = the GPU version in pyexoj_gpu.py (opt-in; validated against this code)."""
    global _POOL_ARGS
    if device:
        from .gpu import detect_spots_gpu
        return detect_spots_gpu(det_stack, p, roi, device=device)
    T = det_stack.shape[0]
    spot_r = jround_f(p.minSize / 2.0)
    if workers > 1 and T > 1:
        import multiprocessing as mp
        _POOL_ARGS = (det_stack, p, roi, fast)
        try:
            with mp.get_context("fork").Pool(min(workers, T)) as pool:
                res = pool.map(_detect_worker, range(T), chunksize=max(1, T // (8 * workers)))
        finally:
            _POOL_ARGS = None
    else:
        res = [_detect_frame(det_stack[t], p, roi, fast) for t in range(T)]
    return [[Spot(int(x), int(y), t + 1, spot_r) for y, x in zip(ys, xs)] for t, (ys, xs) in enumerate(res)]


# ----------------------------------------------------------------------------------- step 2

def _crop_box(x, y, cr, W, H):
    roi_w = 4 * cr + 1
    rx, ry = max(0, x - 2 * cr), max(0, y - 2 * cr)
    if rx + roi_w > W:
        rx = W - roi_w
    if ry + roi_w > H:
        ry = H - roi_w
    return rx, ry, roi_w


def _crop_radius(r):
    return 2 if r < 2 else r + 1


_EXACT_F32 = (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32))


def _frame(img_stack, sl, fast):
    """Frame sl (1-based) as float32 values. fast: the raw frame without a whole-frame copy when its dtype converts
    to float32 exactly and monotonically (uint8/uint16/float32), so every comparison, max and min is unchanged."""
    fr = img_stack[sl - 1]
    if fast and fr.dtype in _EXACT_F32:
        return fr
    return fr.astype(np.float32)


def set_image(s: Spot, img_stack: np.ndarray, fast: bool = False):
    """Vesicle.setImage(ImageStack): move to the disk maximum, crop, max/min over the crop."""
    fr = _frame(img_stack, s.slice, fast)
    H, W = fr.shape
    cx, cy = s.x, s.y
    best = fr[cy, cx]
    r = s.radius
    for j in range(-r, r + 1):
        for i in range(-r, r + 1):
            if i * i + j * j <= r * r:
                v = fr[cy + j, cx + i]
                if v > best:
                    s.x, s.y, best = cx + i, cy + j, v
    cr = _crop_radius(r)
    rx, ry, w = _crop_box(s.x, s.y, cr, W, H)
    s.crop = img_stack[s.slice - 1][ry:ry + w, rx:rx + w].copy()
    c = s.crop.astype(np.float32)
    s.max_den, s.min_den = int(np.trunc(c.max())), int(np.trunc(c.min()))
    s.gfit = None


def expanded_spot(x, y, r, sl, img_stack, fixed, fast: bool = False) -> Spot:
    """Vesicle.createExpandedLocal: max/min over the disk only; optional move to the disk maximum."""
    fr = _frame(img_stack, sl, fast)
    H, W = fr.shape
    if not (0 <= x < W and 0 <= y < H):
        raise ValueError("Expanded-frame center lies outside the source image")
    s = Spot(x, y, sl, r, property="Expanded", is_valid=False)
    mx = mn = fr[y, x]
    for j in range(-r, r + 1):
        for i in range(-r, r + 1):
            if i * i + j * j <= r * r:
                px, py = x + i, y + j
                if 0 <= px < W and 0 <= py < H:
                    v = fr[py, px]
                    if v > mx:
                        mx = v
                        if not fixed:
                            s.x, s.y = px, py
                    if v < mn:
                        mn = v
    cr = _crop_radius(r)
    w = 4 * cr + 1
    if w > W or w > H:
        raise ValueError("Expanded-frame crop is larger than the source image")
    rx, ry, w = _crop_box(s.x, s.y, cr, W, H)
    s.crop = img_stack[sl - 1][ry:ry + w, rx:rx + w].copy()
    s.max_den, s.min_den = int(np.trunc(float(mx))), int(np.trunc(float(mn)))
    return s


def _grid(spots, cell):
    g = {}
    for to, t in enumerate(spots):
        g.setdefault((t.x // cell, t.y // cell), []).append((to, t))
    return g


def track(frames: list[list[Spot]], p: Params, img_stack: np.ndarray, fast: bool = False) -> list[list[Spot]]:
    """Step 2 (TrackingTask, 1.9.2): one-to-one assignment per frame pair, gap closing by depth.
    fast: candidate pairs from a grid of cells >= the search radius (only the 3x3 neighbouring cells can hold a
    target closer than R); the candidates are then sorted on the same full key, so the links are identical."""
    tracks: list[list[Spot]] = []
    owner: dict[int, list[Spot]] = {}
    for fr in frames:
        for s in fr:
            t = [s]
            tracks.append(t)
            owner[id(s)] = t
    nF = len(frames)
    R = p.SpatialsearchRadius
    removed: set[int] = set()
    for gap in range(1, p.TemporalSearchDepth + 1):
        for f in range(0, nF - gap):
            src, tgt = frames[f], frames[f + gap]
            if not src or not tgt:
                continue
            cands = []
            grid = _grid(tgt, max(1, int(math.ceil(R)))) if fast else None
            cell = max(1, int(math.ceil(R)))
            for so, s in enumerate(src):
                st = owner.get(id(s))
                if st is None or st[-1] is not s:
                    continue
                if fast:
                    cx, cy = s.x // cell, s.y // cell
                    near = [c for dy in (-1, 0, 1) for dx in (-1, 0, 1) for c in grid.get((cx + dx, cy + dy), ())]
                else:
                    near = enumerate(tgt)
                for to, t in near:
                    tt = owner.get(id(t))
                    if tt is None or tt[0] is not t or tt is st:
                        continue
                    d = math.hypot(float(s.x - t.x), float(s.y - t.y))
                    if d < R:
                        cands.append((d, s.y, s.x, t.y, t.x, so, to, s, t))
            cands.sort(key=lambda c: c[:7])
            used_s, used_t = set(), set()
            for c in cands:
                s, t = c[7], c[8]
                if id(s) in used_s or id(t) in used_t:
                    continue
                st, tt = owner.get(id(s)), owner.get(id(t))
                if st is None or tt is None or st is tt or st[-1] is not s or tt[0] is not t:
                    continue
                st.extend(tt)
                for v in tt:
                    owner[id(v)] = st
                removed.add(id(tt))       # Java: tracks.remove(targetTrack); order of the rest kept
                used_s.add(id(s))
                used_t.add(id(t))
    tracks = [tr for tr in tracks if id(tr) not in removed]
    tracks.sort(key=lambda tr: (tr[0].slice, tr[0].y, tr[0].x, len(tr)))
    kept = []
    for tr in tracks:
        if sum(1 for v in tr if v.property == "Automatic") >= p.minimalEventSize:
            for v in tr:
                set_image(v, img_stack, fast)
            kept.append(tr)
    return kept


# ----------------------------------------------------------------------------------- step 3

def _simple_regression_slopes(x: np.ndarray, y: np.ndarray, starts: np.ndarray, w: int) -> np.ndarray:
    """Commons Math SimpleRegression(true).getSlope over windows x[s:s+w] (same update formulas)."""
    n = 0
    xbar = x[starts].astype(np.float64)
    ybar = y[starts].astype(np.float64)
    sxx = np.zeros(len(starts))
    sxy = np.zeros(len(starts))
    for j in range(1, w):
        n = j
        xi, yi = x[starts + j], y[starts + j]
        fact1 = 1.0 + n
        fact2 = n / (1.0 + n)
        dx = xi - xbar
        dy = yi - ybar
        sxx = sxx + dx * dx * fact2
        sxy = sxy + dx * dy * fact2
        xbar = xbar + dx / fact1
        ybar = ybar + dy / fact1
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where(np.abs(sxx) < 5.0e-323, np.nan, sxy / sxx)
    if w < 2:
        slope[:] = np.nan
    return slope


def find_peak(curve: np.ndarray, xs: np.ndarray, w: int) -> int:
    """MovingLinearRegretionFinder.dofit + findInflection."""
    T = len(curve)
    sd = np.zeros(T)
    if w >= 2:
        i = np.arange(w - 1, T - w + 1)
        if len(i):
            sl_r = _simple_regression_slopes(xs, curve, i, w)
            sl_l = _simple_regression_slopes(xs, curve, i - w + 1, w)
            sd[i] = sl_l - sl_r
    s = 0.0
    for v in curve:
        s += v
    avg = s / T
    mx, K1 = 0.0, 0
    for i in range(w - 1, T - w + 1):
        if sd[i] > mx and curve[i] > avg:
            mx, K1 = sd[i], i
    if K1 == 0:
        return 0
    mx, K2 = 0.0, 0
    for i in range(K1 - w // 2, K1 + w // 2):
        if curve[i] > mx:
            mx, K2 = curve[i], i
    return K2


def _exp_offset_sse(b, x, y, sumY, sumY2, n):
    """CurveFitter.doRegression for f = exp(-b x), offset + factor by regression."""
    f = np.exp(-b * x)
    sX = sX2 = sXY = 0.0
    for fi, yi in zip(f, y):
        sX += fi
        sX2 += fi * fi
        sXY += fi * yi
    factor = (sXY - sX * sumY / n) / (sX2 - sX * sX / n)
    if not math.isfinite(factor):
        factor = 0.0
    offset = (sumY - factor * sX) / n
    sse = factor * factor * sX2 + n * offset * offset + sumY2 + 2.0 * factor * offset * sX \
        - 2.0 * factor * sXY - 2.0 * offset * sumY
    floor = 2.0e-15 * (factor * factor * sX2 + n * offset * offset + sumY2)
    if sse < floor:
        sse = floor
    return sse, factor, offset


def fit_exp_offset(x: np.ndarray, y: np.ndarray):
    """ImageJ CurveFitter.doFit(EXP_WITH_OFFSET): y = a exp(-b x) + c. Returns a, b, c, R²."""
    n = len(x)
    sumY = sumY2 = 0.0
    for v in y:
        sumY += v
        sumY2 += v * v
    span = (x.max() - x.min()) + 1e-100
    b0, db = 1.0 / span, 0.1 / span
    def g(b):
        with np.errstate(all="ignore"):     # steep b overflows exp(); handled as in ImageJ (factor -> 0)
            return _exp_offset_sse(b, x, y, sumY, sumY2, n)[0]
    try:
        xa, xb, xc, *_ = optimize.bracket(g, b0, b0 + db, maxiter=200)
        res = optimize.minimize_scalar(g, bracket=(xa, xb, xc), method="brent", tol=1e-15)
        b = float(res.x)
    except Exception:                     # no bracket (monotonic SSE): best of a wide scan
        grid = np.concatenate([-np.logspace(-6, 2, 200)[::-1], np.logspace(-6, 3, 400)]) / span
        b = float(grid[np.argmin([g(v) for v in grid])])
    sse, a, c = _exp_offset_sse(b, x, y, sumY, sumY2, n)
    smd = sumY2 - sumY * sumY / n
    r2 = 1.0 - sse / smd if smd > 0.0 else 0.0
    return a, b, c, r2


def gaussian2d_fit(s: Spot) -> dict | None:
    """Vesicle.ensure2DGaussianFit + VesicleFitter (Commons Math LM ≈ MINPACK lmder)."""
    if s.gfit is not None:
        return s.gfit if s.gfit.get("ok") else None
    s.gfit = {"ok": False}
    if s.crop is None:
        return None
    pix = s.crop.astype(np.float32).astype(np.float64).ravel()
    h, w = s.crop.shape
    cr = (w - 1) // 4
    yy, xx = np.mgrid[-2 * cr:2 * cr + 1, -2 * cr:2 * cr + 1]
    mask = (xx * xx + yy * yy <= cr * cr * 4).astype(np.float64).ravel()
    idx = np.arange(pix.size)
    X, Y = idx % w, idx // w
    sel = mask == 1.0
    p0 = np.array([s.min_den, s.max_den - s.min_den, (w - 1) / 2.0, (h - 1) / 2.0, max(1.0, s.radius)], float)

    def model(p, X, Y):
        return p[0] + p[1] * np.exp(-((X - p[2]) ** 2 + (Y - p[3]) ** 2) / (2.0 * p[4] ** 2))

    def resid(p):
        return pix[sel] - model(p, X[sel], Y[sel])

    def jac(p):
        Xs, Ys = X[sel], Y[sel]
        q = model(p, Xs, Ys) - p[0]
        dx, dy = Xs - p[2], Ys - p[3]
        J = np.stack([np.ones_like(q), q / p[1], dx * q / p[4] ** 2, dy * q / p[4] ** 2,
                      (dx * dx + dy * dy) * q / p[4] ** 3], axis=1)
        return -J

    try:
        with np.errstate(all="ignore"):
            pf, cov, info, msg, ier = optimize.leastsq(resid, p0, Dfun=jac, full_output=True, ftol=1e-10,
                                                       xtol=1e-10, gtol=1e-10, maxfev=10000, factor=100)
    except Exception:
        return None
    if ier not in (1, 2, 3, 4) or not np.all(np.isfinite(pf)) or pf[4] == 0.0:
        return None
    fdata = model(pf, X, Y)
    nw = int(sel.sum())
    sumY = sumY2 = ssres = 0.0
    for yi, wi, fi in zip(pix, mask, fdata):
        sumY += yi * wi
        sumY2 += (yi * wi) ** 2
        ssres += ((yi - fi) * wi) ** 2
    if nw <= 0:
        return None
    sst = sumY2 - sumY * sumY / nw
    r2 = 1.0 - ssres / sst if sst > 0.0 else 0.0
    if not math.isfinite(r2):
        return None
    s.gfit = {"ok": True, "params": pf, "r2": r2}
    return s.gfit


@dataclass
class Event:
    ref: int
    spots: list[Spot]
    peak_pos: int = -1
    peak_time: int = -1
    peak_x: int = 0
    peak_y: int = 0
    decay_tau: float = 0.0
    decay_r2: float = 0.0
    decay_peak: float = 0.0
    decay_bk: float = 0.0
    kind: str = "Automatic"          # ExoJ Secretion.proprety

    @property
    def curve(self):
        return np.array([s.max_den for s in self.spots], float)

    @property
    def frames(self):
        return np.array([s.slice for s in self.spots], float)

    def diff1(self):
        c = self.curve
        d = np.zeros_like(c)
        d[1:] = c[1:] - c[:-1]
        return d

    def mad(self):
        d = self.diff1()
        return float(jmedian(np.abs(d - jmedian(d)))) / 0.6745

    def snr(self):
        d = self.diff1()
        res = np.abs(d - jmedian(d))
        mx = 0.0
        for v in res:
            if v > mx:
                mx = v
        m = float(jmedian(res)) / 0.6745
        return 0.0 if m == 0.0 else mx / m

    def max_displacement(self):
        auto = [s for s in self.spots if s.property == "Automatic"]
        if not auto:
            return 0.0
        x0, y0 = auto[0].x, auto[0].y
        m = 0.0
        for s in auto:
            d2 = (s.x - x0) ** 2.0 + (s.y - y0) ** 2.0
            if d2 > m:
                m = d2
        return math.sqrt(m)

    def peak_spot(self):
        return self.spots[self.peak_pos] if 0 <= self.peak_pos < len(self.spots) else None

    def size2d(self):
        pk = self.peak_spot()
        if pk is None:
            return float("nan")
        if not pk.is_valid:
            return 0.0
        g = gaussian2d_fit(pk)
        return float("nan") if g is None else 2.0 * math.sqrt(2.0 * math.log(2.0)) * abs(g["params"][4])

    def size2d_r2(self):
        pk = self.peak_spot()
        if pk is None:
            return float("nan")
        if not pk.is_valid:
            return 0.0
        g = gaussian2d_fit(pk)
        return float("nan") if g is None else g["r2"]


def _fit_decay(ev: Event, min_points: int):
    """Secretion.Fit."""
    if not (min_points > 3 and ev.peak_pos > 0):
        ev.decay_r2 = -2.0
        return
    y, x = ev.curve, ev.frames
    i = ev.peak_pos
    while i < len(y) - 1 and y[i] <= y[i + 1]:
        ev.peak_pos += 1
        ev.peak_time = ev.spots[ev.peak_pos].slice
        i += 1
    if len(x) - ev.peak_pos >= min_points:
        xd = x[ev.peak_pos:] - x[ev.peak_pos]
        yd = y[ev.peak_pos:]
        a, b, c, r2 = fit_exp_offset(xd, yd)
        ev.decay_tau = math.inf if b == 0.0 else 1.0 / b
        ev.decay_bk, ev.decay_peak, ev.decay_r2 = c, a, r2
        ev.peak_x, ev.peak_y = ev.spots[ev.peak_pos].x, ev.spots[ev.peak_pos].y
    else:
        ev.decay_r2 = -1.0


def _copy_spot(s: Spot) -> Spot:
    return Spot(s.x, s.y, s.slice, s.radius, s.property, s.is_valid, s.max_den, s.min_den,
                None if s.crop is None else s.crop, None)


def _identify_one(tr: list[Spot], p: Params, img_stack: np.ndarray, fast: bool, T: int):
    """One track of step 3 (the body of identify's loop): the Event, or None when it is filtered out."""
    spots = [_copy_spot(s) for s in tr]
    try:
        if p.useExpandFrames:
            first, last = spots[0], spots[-1]
            r = first.radius
            sx, sy, ss = first.x, first.y, first.slice
            ex, ey, es = last.x, last.y, last.slice
            for k in range(1, p.expand_frames_L + 1):
                if ss - k >= 1:
                    spots.insert(0, expanded_spot(sx, sy, r, ss - k, img_stack, p.fixedExpand, fast))
            for k in range(1, p.expand_frames_R + 1):
                if es + k <= T:
                    spots.append(expanded_spot(ex, ey, r, es + k, img_stack, p.fixedExpand, fast))
        ev = Event(0, spots)
        if p.useMinSNR and ev.snr() < p.min_SNR:
            return None
        if p.useMaxDisplacement and ev.max_displacement() > p.MaxDisplacement:
            return None
        npts = p.minimalPointsForFitter if p.useMinimalPointsForFitter else 4
        pos = find_peak(ev.curve, ev.frames, npts)
        if 0 <= pos < len(spots):
            ev.peak_pos, ev.peak_time = pos, spots[pos].slice
        _fit_decay(ev, npts)
        if p.useMaxTauFrames and ev.decay_tau > p.max_tau:
            return None
        if p.useMinTauFrames and ev.decay_tau < p.min_tau:
            return None
        if p.useMinDecayFitterR2 and ev.decay_r2 < p.min_decay_fitter_r2:
            return None
        if p.needs_gaussian:
            pk = ev.peak_spot()
            if pk is None or gaussian2d_fit(pk) is None:
                return None
            r2, size = ev.size2d_r2(), ev.size2d()
            if p.useGaussienfitter2DR2 and r2 < p.min_gaussienfitter2DR2:
                return None
            if p.useMaxSize and size > p.max_estimated_size:
                return None
            if p.useMinSize and size < p.min_estimated_size:
                return None
    except (ValueError, IndexError):          # ExoJ: RuntimeException → event skipped
        return None
    return ev


_ID_ARGS = None


def _identify_chunk(bounds):
    tracks, p, img_stack, fast, T = _ID_ARGS
    return [_identify_one(tracks[i], p, img_stack, fast, T) for i in range(*bounds)]


def identify(tracks: list[list[Spot]], p: Params, img_stack: np.ndarray, fast: bool = False,
             workers: int = 1) -> list[Event]:
    """Step 3 (SecretionDetectorTask + performSecretionAnalysis).
    workers > 1: tracks in ordered chunks in forked processes (opt-in). Each track is independent; the results come
    back in track order and get their ref numbers afterwards, as in the one-process loop."""
    global _ID_ARGS
    T = img_stack.shape[0]
    if workers > 1 and len(tracks) >= 4 * workers:
        import multiprocessing as mp
        n = len(tracks)
        k = max(1, n // (16 * workers))
        bounds = [(i, min(n, i + k)) for i in range(0, n, k)]
        _ID_ARGS = (tracks, p, img_stack, fast, T)
        try:
            with mp.get_context("fork").Pool(workers) as pool:
                res = [e for chunk in pool.map(_identify_chunk, bounds, chunksize=1) for e in chunk]
        finally:
            _ID_ARGS = None
    else:
        res = [_identify_one(tr, p, img_stack, fast, T) for tr in tracks]
    out = []
    for ev in res:
        if ev is None:
            continue
        ev.ref = len(out) + 1
        out.append(ev)
    return out


# ----------------------------------------------------------------------------------- driver + output

@dataclass
class Result:
    params: Params
    n_spots: int
    n_tracks: int
    events: list[Event] = field(default_factory=list)


def run(stack: np.ndarray, p: Params, roi: np.ndarray | None = None, workers: int = 1, fast: bool = False,
        device: str | None = None) -> Result:
    """Steps 1–3 on a (T, Y, X) single-channel stack (8/16-bit integer or float32).
    workers/fast: opt-in speed-ups (identical results, see README): step 1 (workers, fast), and with any truthy
    fast also steps 2-3 (grid candidate search, no whole-frame float copies)."""
    if stack.ndim != 3:
        raise ValueError("need a (T, Y, X) stack")
    img = bleaching_correction(stack) if p.bleach_correct else stack
    det = abs_delta_movie(img) if p.deltamovie else img
    frames = detect_spots(det, p, roi, workers=workers, fast=fast, device=device)
    tracks = track(frames, p, img, fast=bool(fast))
    events = identify(tracks, p, img, fast=bool(fast), workers=workers)
    return Result(p, sum(len(f) for f in frames), len(tracks), events)


def java_number(x: float) -> str:
    """java.text.NumberFormat.getNumberInstance(Locale.US).format(double): ≤3 decimals, HALF_EVEN, grouping."""
    if isinstance(x, float) and math.isnan(x):
        return "NaN"
    if isinstance(x, float) and math.isinf(x):
        return "∞" if x > 0 else "-∞"
    d = Decimal(float(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_EVEN)
    if d == 0:
        d = abs(d)
    s = f"{d:,f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


TABLE_COLUMNS = ["Ref", "Property", "Begin Frame", "Duration", "Peak Frame", "Position (x,y)",
                 "Max. Displacement (pixels)", "τ (sec)", "R² for Decay Est.", "dF/σ (MAD)", "F0", "ΔF",
                 "R² for Gaussian fit", "Size (FWHM, microns)"]


def table_rows(res: Result) -> list[list[str]]:
    """The GUI's result table (ResultTableWin.populateTable), as strings."""
    p = res.params
    rows = []
    for e in res.events:
        g = gaussian2d_fit(e.peak_spot()) if (p.needs_gaussian and e.peak_spot() is not None) else None
        if e.peak_spot() is not None and e.peak_spot().gfit is not None and e.peak_spot().gfit.get("ok"):
            g = e.peak_spot().gfit
        r2s = java_number(e.size2d_r2()) if g else "N/A"
        sz = java_number(e.size2d() * p.pixelSize) if g else "N/A"
        c = e.curve
        rows.append([str(e.ref), e.kind, str(e.spots[0].slice), str(len(e.spots)), str(e.peak_time),
                     f"{e.spots[0].x},{e.spots[0].y}", java_number(e.max_displacement()),
                     java_number(e.decay_tau * p.timeInterval), java_number(e.decay_r2), java_number(e.snr()),
                     java_number(c[0]), java_number(c.max() - c[0]), r2s, sz])
    return rows


def write_outputs(res: Result, out_dir, name: str):
    import os
    os.makedirs(out_dir, exist_ok=True)
    p = res.params
    with open(os.path.join(out_dir, f"{name}_exoj_table.tsv"), "w") as f:
        f.write("\t".join(TABLE_COLUMNS) + "\n")
        for r in table_rows(res):
            f.write("\t".join(r) + "\n")
    with open(os.path.join(out_dir, f"{name}_events.tsv"), "w") as f, \
            open(os.path.join(out_dir, f"{name}_traces.tsv"), "w") as tr:
        f.write("ref\tproperty\tpeak_x\tpeak_y\tpeak_frame\tstart_frame\tfin_frame\tstart_x\tstart_y\tn_points"
                "\tdecay_tau_frames\tdecay_r2\tdecay_peak\tdecay_bk\tF0\tdeltaF\tSNR\tMAD\tmax_displacement_px"
                "\tsize2D\tsize2D_r2\n")
        tr.write("ref\tindex\tframe\tx\ty\tproperty\tintensity\n")
        for e in res.events:
            c = e.curve
            size, sr2 = (e.size2d(), e.size2d_r2()) if p.needs_gaussian else (float("nan"), float("nan"))
            vals = [e.ref, e.kind, e.peak_x, e.peak_y, e.peak_time, e.spots[0].slice, e.spots[-1].slice,
                    e.spots[0].x, e.spots[0].y, len(e.spots), e.decay_tau, e.decay_r2, e.decay_peak, e.decay_bk,
                    c[0], c.max() - c[0], e.snr(), e.mad(), e.max_displacement(), size, sr2]
            f.write("\t".join(_jstr(v) for v in vals) + "\n")
            for i, s in enumerate(e.spots):
                tr.write(f"{e.ref}\t{i}\t{s.slice}\t{s.x}\t{s.y}\t{s.property}\t{_jstr(float(s.max_den))}\n")
    with open(os.path.join(out_dir, f"{name}_params_used.tsv"), "w") as f:
        f.write("field\tvalue\n")
        for k, v in asdict(p).items():
            f.write(f"{k}\t{v}\n")
        f.write(f"scaleList\t{p.scale_list()}\n")


def _jstr(v) -> str:
    """Java's Double.toString-like rendering for comparison files (repr for floats)."""
    if isinstance(v, (np.integer,)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        return repr(v)
    return str(v)
