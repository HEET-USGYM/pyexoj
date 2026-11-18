import hashlib
import os

import numpy as np

from pyexoj import cli
from pyexoj.synthetic import make_movie

DATA = os.path.join(os.path.dirname(__file__), "data")
CASES = ["A", "B1", "B2", "B3"]
# sha256 of make_movie() as used to produce tests/data/exoj_1.9.2 (ExoJ 1.9.2 via tools/exoj_cli)
MOVIE_SHA256 = "e02150659a8ecaf89df9c420939afd27441c610e29802135628d5ef3174b3486"
_movie = None


def movie():
    global _movie
    if _movie is None:
        _movie = make_movie()[0]
    return _movie


def movie_sha256():
    return hashlib.sha256(np.ascontiguousarray(movie()).tobytes()).hexdigest()


def params(case):
    return cli.params_from(cli.read_props(os.path.join(DATA, "settings", f"{case}.properties")))


def read(path):
    with open(path, "rb") as f:
        return f.read()
