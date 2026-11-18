"""Command line: run pyexoj on one stack with an ExoJCli-style .properties settings file (same keys).

    pyexoj settings.properties [--out DIR] [--workers N] [--fast {false,true,inplace}] [--device cuda:0]
"""
import argparse
import os
import time

from . import core as px

MAP = {  # settings key -> (Params field, type)
    "pixel_size_um": ("pixelSize", float), "frame_interval_s": ("timeInterval", float),
    "min_size": ("minSize", int), "max_size": ("maxSize", int), "snr_vesicle": ("SNR_Vesicle", float),
    "wavelet": ("WaveletFilter", "b"), "bleach_correct": ("bleach_correct", "b"), "delta_movie": ("deltamovie", "b"),
    "search_radius": ("SpatialsearchRadius", int), "temporal_depth": ("TemporalSearchDepth", int),
    "min_event_size": ("minimalEventSize", int),
    "use_min_points": ("useMinimalPointsForFitter", "b"), "min_points": ("minimalPointsForFitter", int),
    "use_expand": ("useExpandFrames", "b"), "expand_pre": ("expand_frames_L", int), "expand_post": ("expand_frames_R", int),
    "fixed_expand": ("fixedExpand", "b"), "use_min_snr": ("useMinSNR", "b"), "min_snr": ("min_SNR", float),
    "use_max_tau": ("useMaxTauFrames", "b"), "max_tau": ("max_tau", float),
    "use_min_tau": ("useMinTauFrames", "b"), "min_tau": ("min_tau", float),
    "use_min_decay_r2": ("useMinDecayFitterR2", "b"), "min_decay_r2": ("min_decay_fitter_r2", float),
    "use_max_size": ("useMaxSize", "b"), "max_est_size": ("max_estimated_size", float),
    "use_min_size": ("useMinSize", "b"), "min_est_size": ("min_estimated_size", float),
    "use_min_size_r2": ("useGaussienfitter2DR2", "b"), "min_size_r2": ("min_gaussienfitter2DR2", float),
    "use_max_displacement": ("useMaxDisplacement", "b"), "max_displacement": ("MaxDisplacement", float),
}


def read_props(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
    return d


def params_from(d):
    p = px.Params()
    for k, v in d.items():
        if k in MAP:
            f, t = MAP[k]
            setattr(p, f, (v.lower() == "true") if t == "b" else t(v))
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pyexoj", description=__doc__.splitlines()[0])
    ap.add_argument("settings", help="ExoJCli-style .properties file (input=, out_dir=, name=, ExoJ keys)")
    ap.add_argument("--out", help="output folder (default: out_dir from the settings)")
    ap.add_argument("--workers", type=int, default=1, help="processes for steps 1 and 3 (results identical)")
    ap.add_argument("--fast", default="false", choices=["false", "true", "inplace"],
                    help="exact speed-ups (results identical); 'inplace' is the fastest")
    ap.add_argument("--device", help="GPU for step 1, e.g. cuda:0 (needs PyTorch; results identical, validate first)")
    a = ap.parse_args(argv)
    import tifffile
    d = read_props(a.settings)
    out = a.out or d["out_dir"]
    name = d.get("name", os.path.splitext(os.path.basename(d["input"]))[0])
    stack = tifffile.imread(d["input"])
    if stack.ndim == 2:
        stack = stack[None]
    p = params_from(d)
    fast = {"true": True, "false": False, "inplace": "inplace"}[a.fast]
    t0 = time.time()
    res = px.run(stack, p, workers=a.workers, fast=fast, device=a.device)
    px.write_outputs(res, out, name)
    msg = f"{name}: spots={res.n_spots} tracks={res.n_tracks} events={len(res.events)} ({time.time()-t0:.1f} s)"
    with open(os.path.join(out, f"{name}_log.txt"), "w") as f:
        f.write(msg + "\n")
    print(msg)


if __name__ == "__main__":
    main()
