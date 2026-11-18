"""Compare pyexoj output with ExoJ (ExoJCli) output for one run name: table, events, traces.

Usage: python tools/compare.py <exoj_out_dir> <pyexoj_out_dir> <name>
"""
import sys, os
import numpy as np, pandas as pd

def compare(java_dir, py_dir, name, rtol=1e-5):
    rep = {}
    jl = open(os.path.join(java_dir, f"{name}_log.txt")).read()
    pl = open(os.path.join(py_dir, f"{name}_log.txt")).read()
    import re
    g = lambda s, k: int(re.search(k + r"=(\d+)", s).group(1)) if re.search(k + r"=(\d+)", s) else None
    rep["spots"] = (g(jl, "spots"), g(pl, "spots"))
    rep["tracks"] = (g(jl, "tracks"), g(pl, "tracks"))
    rep["events"] = (g(jl, "events"), g(pl, "events"))
    jt = open(os.path.join(java_dir, f"{name}_exoj_table.tsv")).read().splitlines()
    pt = open(os.path.join(py_dir, f"{name}_exoj_table.tsv")).read().splitlines()
    nd = sum(1 for a, b in zip(jt[1:], pt[1:]) if a != b) + abs(len(jt) - len(pt))
    rep["table_rows_differing"] = nd
    rep["table_diff_examples"] = [(a, b) for a, b in zip(jt[1:], pt[1:]) if a != b][:3]
    je = pd.read_csv(os.path.join(java_dir, f"{name}_events.tsv"), sep="\t")
    pe = pd.read_csv(os.path.join(py_dir, f"{name}_events.tsv"), sep="\t")
    if len(je) == len(pe):
        ints = ["ref", "peak_x", "peak_y", "peak_frame", "start_frame", "fin_frame", "start_x", "start_y", "n_points"]
        rep["int_cols_equal"] = bool((je[ints].values == pe[ints].values).all())
        fl = [c for c in je.columns if c not in ints + ["property"]]
        a, b = je[fl].astype(float).values, pe[fl].astype(float).values
        both_nan = np.isnan(a) & np.isnan(b)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.where(both_nan, 0, np.abs(a - b) / np.maximum(np.abs(a), 1e-12))
        worst = np.nanargmax(rel)
        rep["max_rel_diff"] = float(np.nanmax(rel))
        rep["worst"] = (fl[worst % len(fl)], float(a.ravel()[worst]), float(b.ravel()[worst]))
        rep["n_values_over_rtol"] = int((rel > rtol).sum())
    jtr = pd.read_csv(os.path.join(java_dir, f"{name}_traces.tsv"), sep="\t")
    ptr = pd.read_csv(os.path.join(py_dir, f"{name}_traces.tsv"), sep="\t")
    rep["traces_equal"] = len(jtr) == len(ptr) and bool((jtr.values == ptr.values).all())
    return rep

if __name__ == "__main__":
    for k, v in compare(*sys.argv[1:4]).items():
        print(f"{k}: {v}")
