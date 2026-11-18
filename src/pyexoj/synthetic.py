"""Deterministic synthetic TIRF movies with known exocytosis-like events (for tests and examples).

Uses numpy's legacy RandomState, whose streams are frozen across numpy versions. Events: a spot appears within
one frame, then fades exponentially (tau frames) while it widens (sigma grows), on a smooth background with a
few static and moving spots (non-events), Poisson + Gaussian camera noise, uint16.
"""
import numpy as np


def make_movie(seed=1, T=160, H=96, W=112, n_events=10, n_static=4, n_moving=3, background=400.0,
               read_noise=6.0):
    rs = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    bg = background * (1.0 + 0.15 * (xx / W) - 0.10 * (yy / H))
    sig = np.zeros((T, H, W))
    m = 12
    truth = []
    taken = []

    def place():
        while True:
            x, y = rs.uniform(m, W - m), rs.uniform(m, H - m)
            if all((x - a) ** 2 + (y - b) ** 2 > 12 ** 2 for a, b in taken):
                taken.append((x, y))
                return x, y

    def spot(x, y, s, a):
        return a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * s * s))

    for i in range(n_events):
        x, y = place()
        t0 = int(rs.randint(10, T - 40))
        tau = float(rs.uniform(3.0, 15.0))
        amp = float(rs.uniform(250.0, 600.0))
        for t in range(t0, T):
            k = t - t0
            s = 1.4 + 0.06 * k
            sig[t] += spot(x, y, s, amp * np.exp(-k / tau) * (1.4 / s) ** 2)
        truth.append(dict(kind="event", x=round(x, 2), y=round(y, 2), onset_frame=t0 + 1, tau_frames=round(tau, 3),
                          amplitude=round(amp, 1)))
    for i in range(n_static):
        x, y = place()
        a = float(rs.uniform(80.0, 200.0))
        sig += spot(x, y, 1.5, a)[None]
        truth.append(dict(kind="static", x=round(x, 2), y=round(y, 2)))
    for i in range(n_moving):
        x, y = place()
        vx, vy = rs.uniform(-0.6, 0.6, size=2)
        a = float(rs.uniform(150.0, 300.0))
        for t in range(T):
            sig[t] += spot(np.clip(x + vx * t, 3, W - 4), np.clip(y + vy * t, 3, H - 4), 1.5, a)
        truth.append(dict(kind="moving", x=round(x, 2), y=round(y, 2)))
    lam = bg[None] + sig
    movie = rs.poisson(lam).astype(np.float64) + rs.normal(0.0, read_noise, size=lam.shape)
    movie = np.clip(np.rint(movie), 0, 65535).astype(np.uint16)
    return movie, truth
