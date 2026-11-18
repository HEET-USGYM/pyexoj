"""pyexoj vs ExoJ 1.9.2 (outputs of the original jar, made with tools/exoj_cli on the synthetic movie)."""
import math
import os

import numpy as np
import pandas as pd

import pyexoj as px
from pyexoj import core
from _common import CASES, DATA, MOVIE_SHA256, movie, movie_sha256, params

REF = os.path.join(DATA, "exoj_1.9.2")
INT_COLS = ["ref", "peak_x", "peak_y", "peak_frame", "start_frame", "fin_frame", "start_x", "start_y", "n_points"]


def _decay_sse(trace, peak_frame, tau):
    """Residual error of the decay fit y = a exp(-x/tau) + c from the peak on (as ImageJ / pyexoj fit it)."""
    tr = trace[trace.frame >= peak_frame]
    x = (tr.frame.values - peak_frame).astype(float)
    y = tr.intensity.values.astype(float)
    b = 1.0 / tau if tau not in (0.0,) and math.isfinite(tau) else 0.0
    return core._exp_offset_sse(b, x, y, float(y.sum()), float((y * y).sum()), len(y))[0], float(((y - y.mean()) ** 2).sum())


def check_case(case, tmp):
    res = px.run(movie(), params(case))
    px.write_outputs(res, tmp, case)
    je = pd.read_csv(os.path.join(REF, f"{case}_events.tsv"), sep="\t")
    pe = pd.read_csv(os.path.join(tmp, f"{case}_events.tsv"), sep="\t")
    log = open(os.path.join(REF, f"{case}_log.txt")).read()
    assert f"spots={res.n_spots}" in log and f"tracks={res.n_tracks}" in log and f"events={len(res.events)}" in log
    assert len(je) == len(pe)
    assert (je[INT_COLS].values == pe[INT_COLS].values).all(), "integer columns differ"
    jt = pd.read_csv(os.path.join(REF, f"{case}_traces.tsv"), sep="\t")
    pt = pd.read_csv(os.path.join(tmp, f"{case}_traces.tsv"), sep="\t")
    assert jt.shape == pt.shape and (jt.values == pt.values).all(), "traces differ"
    # fitted values: equal within 1e-5, or (decay fit only) a flat minimum: same residual error at both taus
    flat = 0
    for i in range(len(je)):
        for col in ["decay_tau_frames", "decay_r2", "decay_peak", "decay_bk", "F0", "deltaF", "SNR", "MAD",
                    "max_displacement_px", "size2D", "size2D_r2"]:
            a, b = float(je[col].iloc[i]), float(pe[col].iloc[i])
            if (math.isnan(a) and math.isnan(b)) or a == b or abs(a - b) <= 1e-5 * max(abs(a), 1e-12):
                continue
            assert col.startswith("decay_"), f"{case} event {i + 1}: {col} {a} vs {b}"
            tr = pt[pt.ref == pe.ref.iloc[i]]
            sj, sst = _decay_sse(tr, pe.peak_frame.iloc[i], float(je.decay_tau_frames.iloc[i]))
            sp, _ = _decay_sse(tr, pe.peak_frame.iloc[i], float(pe.decay_tau_frames.iloc[i]))
            assert sp <= sj + 1e-6 * sst, f"{case} event {i + 1}: pyexoj fit error {sp} > ExoJ {sj}"
            flat += 1
    return flat


def test_movie_is_the_reference_movie():
    assert movie_sha256() == MOVIE_SHA256, "synthetic movie differs from the one ExoJ was run on (numpy/libm?)"


def test_cases(tmp_path):
    if movie_sha256() != MOVIE_SHA256:
        import pytest
        pytest.skip("synthetic movie differs from the reference movie on this platform")
    for case in CASES:
        d = tmp_path / case
        d.mkdir()
        check_case(case, str(d))
