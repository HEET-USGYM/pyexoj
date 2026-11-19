# pyexoj

An **unofficial** Python port of [ExoJ](https://github.com/zs6e/excytosis-analyzer-plugin) 1.9.2, the
ImageJ2/Fiji plugin for automated detection and analysis of exocytosis events in TIRF time series.

pyexoj runs the same three steps as ExoJ 1.9.2 (spot detection, tracking, event identification) without
Fiji or a GUI, gives the same spots, tracks and events, and adds exact speed-ups: on a dense 2048² movie,
925 s → 79 s on the CPU alone, and 18.6 s with spot detection on a GPU, with byte-identical results.
It is not affiliated with or endorsed by the ExoJ authors. **If you use it, cite ExoJ:**

> Liu J., Verweij F.J., van Niel G., Galli T., Danglot L., Bun P. (2024). ExoJ: an ImageJ2/Fiji plugin for
> automated spatiotemporal detection and analysis of exocytosis. *J. Cell Sci.* 137(20): jcs261938.
> https://doi.org/10.1242/jcs.261938

## Install

```bash
pip install .            # numpy + scipy
pip install ".[tiff]"    # + tifffile, for the command line tool
pip install ".[gpu]"     # + PyTorch (CUDA or ROCm build), for step 1 on a GPU
```

## Use

```python
import tifffile, pyexoj as px
p = px.Params(pixelSize=0.065, timeInterval=0.5, minSize=3, maxSize=7, expand_frames_R=20)
res = px.run(tifffile.imread("movie.tif"), p)       # (T, Y, X) stack, 8/16-bit or float32
px.write_outputs(res, "out/", "movie")              # ExoJ's result table + events, traces, settings
```

`Params` uses ExoJ's own field names and defaults. From the command line, with the settings keys of
`tools/exoj_cli/README.md`:

```bash
pyexoj settings.properties --workers 8 --fast inplace [--device cuda:0]
```

Note: ExoJ's step-1 sizes (`minSize`, `maxSize`) are apparent spot sizes in **pixels** (1–25, default 4–8),
not radii. As in ExoJ, the movie should be drift-corrected first.

## What is ported

Optional photobleaching detrend and |ΔF| movie → per-frame à-trous (B3) wavelet detection or normalised
local maxima → one-to-one frame-to-frame tracking with gap closing → per track: frame expansion, ΔF/σ(MAD)
and displacement filters, moving-regression peak, decay fit y = a·e^(−x/τ) + c with R², optional 2-D
Gaussian size (FWHM) with limits, and ExoJ's result table. Not ported: the GUI and its views, ring-sampling
export, and ExoJ's 1-D Gaussian (not used by its pipeline). An optional `roi` is a boolean (Y, X) mask.

## Agreement with ExoJ 1.9.2

`tests/test_exoj_reference.py` runs pyexoj on a deterministic synthetic movie (`pyexoj.synthetic`) with
four settings sets (wavelet and local-maxima detection, bleaching correction, |ΔF| movie, gap closing 1–3,
fixed and moving expansion, τ limits, size fit and limits, displacement limit) and compares with the outputs
of the original ExoJ 1.9.2 jar on the same movie (`tests/data/exoj_1.9.2/`, made with `tools/exoj_cli`):

- spot, track and event counts, all integer columns (positions, frames) and every trace value: **identical**;
- fitted values: equal within 1e-5 (in practice ≤ 3e-6), except decay fits at flat minima, where ImageJ's
  randomised simplex and pyexoj's Brent search stop at different τ with the same residual error (checked:
  pyexoj's error is never larger). These are fades shorter than a frame or much longer than the trace,
  where τ is not determined in either program. ExoJ itself varies by ~1e-6 between runs.

Steps 1 and 2 reproduce ExoJ's float32/float64 arithmetic in the same order. The port was also checked on
real TIRF movies (same spots, tracks and events as ExoJ).

## Speed: up to ~50× faster, with identical results

On a dense movie (2048 × 2048 pixels, 601 frames, ~1,150 spots per frame, 8,016 events; Ryzen 7 7700X,
Radeon RX 6800 XT):

| Run | Time | Speed-up |
|---|---|---|
| pyexoj reference code (one process, operations in ExoJ's order) | 925 s | 1× |
| `workers=8, fast="inplace"` (CPU only) | **79 s** | **12×** |
| `workers=8, fast="inplace", device="cuda:0"` (spot detection on the GPU) | **18.6 s** | **~50×** |

Every option gives **byte-identical output files** to the reference code: the same spots, tracks, events,
fit values and traces, not "nearly the same". What each option does:

| Option | What it does |
|---|---|
| `workers=8` | frames (step 1) and tracks (step 3) in parallel processes |
| `fast="inplace"` | wavelet smoothing in preallocated buffers; reads each spot's pixels without copying the whole frame; looks for link candidates on a grid instead of comparing every spot pair |
| `device="cuda:0"` | step 1 on a GPU (PyTorch, CUDA or ROCm), with float32 arithmetic in the reference's order and exact medians |

(ExoJ itself was not timed on this movie; on small synthetic movies the reference code is somewhat faster
than ExoJ run through Fiji.) In the 18.6 s, detection takes 7.7 s, tracking 4.7 s and events 6.2 s; at that point reading the movie from
disk takes about as long as analysing it. More than 8 workers did not help on that 8-core CPU.
`tests/test_exactness.py` checks every option against the reference code. **Run it on your own GPU and driver
before relying on the GPU path** (checked on Radeon RX 9070 XT and RX 6800 XT, ROCm). The GPU path covers
wavelet detection (ExoJ's default); other detection settings run on the CPU. With `workers > 1`, don't run other
threads in the calling process (it uses fork).

## Tests

```bash
pip install ".[test]" && pytest tests/        # or: python tests/run_tests.py
```

## How this port was made

The ExoJ source on GitHub is older than the released 1.9.2 jar (Zenodo), so pyexoj was written from the
1.9.2 jar decompiled with Vineflower (checked: the recompiled jar behaves like the original), together with
ImageJ 1.54p `CurveFitter` (fit type 11) and Apache Commons Math 3.6.1 (`SimpleRegression`, Levenberg–
Marquardt). The decompiled Java is not distributed here. `tools/exoj_cli/` is a small headless driver for the
original jar (it drives ExoJ's own wizard), used to make the reference outputs.

The code was written with the help of an AI coding assistant (Claude, Anthropic) and verified against the
original program as described above.

## Licence

GPL-3.0-or-later (`LICENSE`), as a derived work of ExoJ.
ExoJ: Copyright (C) 2022 Liu Junjun, Bun Philippe. pyexoj: Copyright (C) 2026 HEET-USGYM.
