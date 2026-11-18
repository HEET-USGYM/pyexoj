"""Sanity: the planted events of the synthetic movie are found (settings A)."""
import numpy as np

import pyexoj as px
from pyexoj.synthetic import make_movie
from _common import params


def test_planted_events_found():
    m, truth = make_movie()
    res = px.run(m, params("A"))
    ev = [t for t in truth if t["kind"] == "event"]
    found = sum(any(np.hypot(e.spots[0].x - t["x"], e.spots[0].y - t["y"]) <= 3 and abs(e.peak_time - t["onset_frame"]) <= 3
                    for e in res.events) for t in ev)
    assert found >= len(ev) - 1, f"only {found}/{len(ev)} planted events found"
