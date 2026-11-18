"""The opt-in speed-ups (workers, fast, device) must give byte-identical outputs to the reference code."""
import os

import pyexoj as px
from _common import CASES, movie, params, read


def _outputs(tmp, tag, case, **kw):
    d = os.path.join(tmp, tag)
    px.write_outputs(px.run(movie(), params(case), **kw), d, case)
    return {f: read(os.path.join(d, f)) for f in sorted(os.listdir(d))}


def test_cpu_speedups(tmp_path):
    for case in CASES:
        ref = _outputs(str(tmp_path), f"{case}_ref", case)
        for kw in [dict(fast=True), dict(fast="inplace"), dict(workers=4), dict(workers=4, fast="inplace")]:
            got = _outputs(str(tmp_path), f"{case}_{kw}", case, **kw)
            assert got == ref, f"{case} {kw}: outputs differ"


def _gpu():
    try:
        import torch
    except ImportError:
        return None
    return "cuda:0" if torch.cuda.is_available() else None


def test_gpu_detection(tmp_path):
    dev = _gpu()
    if dev is None:
        try:
            import pytest
            pytest.skip("no PyTorch GPU")
        except ImportError:
            print("skipped: no PyTorch GPU")
            return
    import numpy as np
    from pyexoj import core, gpu
    m = movie()
    for case in CASES:
        p = params(case)
        if not p.WaveletFilter:
            continue                                   # GPU path covers wavelet detection only
        ref = core.detect_spots(m, p)
        got = gpu.detect_spots_gpu(m, p, device=dev)
        assert [[(s.x, s.y) for s in f] for f in ref] == [[(s.x, s.y) for s in f] for f in got], case
        for t in (0, len(m) // 2):
            a = core.wavelet_filter(m[t], p.scale_list(), p.SNR_Vesicle)
            b = gpu.wavelet_filter_gpu(m[t], p.scale_list(), p.SNR_Vesicle, dev)
            assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), f"{case} frame {t}: wavelet image differs"
        r1 = _outputs(str(tmp_path), f"{case}_cpu", case)
        r2 = _outputs(str(tmp_path), f"{case}_gpu", case, device=dev)
        assert r1 == r2, f"{case}: GPU outputs differ"
