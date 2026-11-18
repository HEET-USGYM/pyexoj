"""pyexoj step 1 (spot detection) on a GPU (PyTorch, ROCm or CUDA), meant to give exactly the spots of
pyexoj.detect_spots (the CPU reference). Opt-in: pyexoj.detect_spots(..., device="cuda:0") or pyexoj.run(..., device=...).

Same arithmetic as pyexoj.wavelet_filter / _detect_frame, step by step:
  - frame -> float32 (exact for uint8/uint16/float32 input);
  - B3 à-trous smoothing as separate element-wise float32 kernels in the reference's order
    W2*(x[-2s]+x[+2s]) + W1*(x[-s]+x[+s]) + W0*x, mirror borders from the reference's index tables
    (no convolution routines, no torch.compile: those change the summation order / fuse into FMAs);
  - per scale: jmedian = (n//2)-th smallest via torch.kthvalue (exact selection), MAD in float32 as in the
    reference (division done in numpy on the returned scalars), threshold lam*mad*k in float64; the float64
    comparison d >= thr is done in float32 against the smallest float32 >= thr (equivalent, no fp64 needed);
  - scales summed in the reference's order; local maxima over the disk footprint (max of shifted views,
    -inf outside), not below 1.0, border margin 2*max_r, optional ROI; coordinates in row-major order.
Only the wavelet path (p.WaveletFilter = True, ExoJ's default) runs here; otherwise the CPU code is used.
Validate against the CPU reference on your own GPU and driver before relying on it (tests/test_exactness.py).
"""
import math

import numpy as np
import torch

from . import core as px

_W2, _W1, _W0 = 0.0625, 0.25, 0.375          # exact in float32 (= pyexoj _W2/_W1/_W0)


def _f32_ceil(thr: float) -> np.float32:
    """Smallest float32 t with t >= thr, so that (float32 d) >= t  <=>  float64(d) >= thr."""
    if math.isnan(thr):
        return np.float32("nan")
    t = np.float32(thr)
    if float(t) < thr:
        t = np.nextafter(t, np.float32(np.inf))
    return t


class _Mirror:
    """Index tables of pyexoj._reflect_index per (length, offset), cached on the device."""

    def __init__(self, device):
        self.device, self.cache = device, {}

    def idx(self, n, off):
        k = (n, off)
        if k not in self.cache:
            self.cache[k] = torch.from_numpy(px._reflect_index(n, off).astype(np.int64)).to(self.device)
        return self.cache[k]


def _smooth(a, step, dim, mir):
    n = a.shape[dim]
    t = lambda off: torch.index_select(a, dim, mir.idx(n, off))
    x = t(-2 * step).add_(t(2 * step)).mul_(_W2)
    y = t(-step).add_(t(step)).mul_(_W1)
    x.add_(y)
    del y
    return x.add_(a * _W0)


def _kth(x2d, k):
    """k-th smallest (1-based) per row; returns a float32 numpy array."""
    return torch.kthvalue(x2d, k, dim=1).values.cpu().numpy().astype(np.float32)


def wavelet_filter_batch(a, scale_list, k, mir):
    """pyexoj.wavelet_filter on a (B, H, W) float32 batch."""
    B, H, W = a.shape
    max_scale = 0
    for i, s in enumerate(scale_list):
        if s:
            max_scale = i + 1
    out = torch.zeros_like(a)
    if max_scale == 0:
        return out
    min_size = 5 + (int(2.0 ** (max_scale - 1)) - 1) * 4
    if W < min_size or H < min_size:
        return out
    npix = W * H
    kk = npix // 2 + 1
    a_prev = a
    for depth in range(max_scale):
        step = int(2.0 ** depth)
        sm = _smooth(_smooth(a_prev, step, 2, mir), step, 1, mir)
        if scale_list[depth]:
            d = a_prev - sm
            q = npix // (1 << (2 * (depth + 1)))
            lam = math.sqrt(2.0 * math.log(q)) if q > 0 else float("nan")
            med = _kth(d.reshape(B, -1), kk)
            dev = (d - torch.from_numpy(med).to(a.device)[:, None, None]).abs_()
            madv = _kth(dev.reshape(B, -1), kk)
            del dev
            mad = (madv / np.float32(0.6745)).astype(np.float32)
            thr = np.array([_f32_ceil(lam * float(m) * k) for m in mad], np.float32)
            keep = d >= torch.from_numpy(thr).to(a.device)[:, None, None]
            d.masked_fill_(~keep, 0.0)                     # +0.0 like the reference
            out = out + d                                  # reference: v = 0; v = v + coeffs[j] in order
            del d, keep
        a_prev = sm
    return out


def _disk_offsets(r):
    m = px._disk(r)
    return [(j - r, i - r) for j in range(2 * r + 1) for i in range(2 * r + 1) if m[j, i]]


def _local_max(img, r):
    """scipy maximum_filter(img, footprint=disk(r), mode='constant', cval=-inf) on a (B, H, W) batch."""
    B, H, W = img.shape
    pad = torch.full((B, H + 2 * r, W + 2 * r), float("-inf"), dtype=img.dtype, device=img.device)
    pad[:, r:r + H, r:r + W] = img
    m = None
    for dy, dx in _disk_offsets(r):
        v = pad[:, r + dy:r + dy + H, r + dx:r + dx + W]
        m = v.clone() if m is None else torch.maximum(m, v, out=m)
    return m


def _to_gpu_f32(block: np.ndarray, device):
    if block.dtype == np.uint16:
        t = torch.from_numpy(np.ascontiguousarray(block).view(np.int16)).to(device)
        return (t.to(torch.int32) & 0xFFFF).to(torch.float32)
    if block.dtype in (np.uint8, np.float32):
        return torch.from_numpy(np.ascontiguousarray(block)).to(device).to(torch.float32)
    return torch.from_numpy(np.ascontiguousarray(block).astype(np.float32)).to(device)


def detect_spots_gpu(det_stack: np.ndarray, p, roi=None, device="cuda:0", batch_pixels=48 * 2**20):
    """Same output as pyexoj.detect_spots (list per frame of Spot, row-major)."""
    if not p.WaveletFilter:
        return px.detect_spots(det_stack, p, roi)
    dev = torch.device(device)
    mir = _Mirror(dev)
    T, H, W = det_stack.shape
    spot_r = px.jround_f(p.minSize / 2.0)
    max_r = px.jround_f(p.maxSize / 2.0)
    y0, y1, x0, x1 = 2 * max_r, H - 2 * max_r, 2 * max_r, W - 2 * max_r
    sl = p.scale_list()
    roi_t = None if roi is None else torch.from_numpy(np.asarray(roi, bool)).to(dev)
    B = max(1, min(T, batch_pixels // (H * W)))
    frames = []
    with torch.no_grad():
        for b0 in range(0, T, B):
            a = _to_gpu_f32(det_stack[b0:b0 + B], dev)
            img = wavelet_filter_batch(a, sl, p.SNR_Vesicle, mir)
            del a
            mxf = _local_max(img, spot_r)
            cand = (img >= mxf) & ~(img < 1.0)
            del mxf, img
            sub = torch.zeros_like(cand)
            if y1 > y0 and x1 > x0:
                sub[:, y0:y1, x0:x1] = cand[:, y0:y1, x0:x1]
            if roi_t is not None:
                sub &= roi_t
            nz = torch.nonzero(sub).cpu().numpy()        # (b, y, x), row-major per frame
            del cand, sub
            nb = min(B, T - b0)
            bnd = np.searchsorted(nz[:, 0], np.arange(nb + 1)) if len(nz) else np.zeros(nb + 1, int)
            for i in range(nb):
                rows = nz[bnd[i]:bnd[i + 1]]
                frames.append([px.Spot(int(x), int(y), b0 + i + 1, spot_r) for _, y, x in rows])
    return frames


def wavelet_filter_gpu(frame: np.ndarray, scale_list, k, device="cuda:0") -> np.ndarray:
    """One frame, for comparisons with pyexoj.wavelet_filter."""
    dev = torch.device(device)
    with torch.no_grad():
        a = _to_gpu_f32(frame[None], dev)
        return wavelet_filter_batch(a, scale_list, k, _Mirror(dev))[0].cpu().numpy()
