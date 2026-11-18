# ExoJCli — ExoJ without the GUI

Runs the ExoJ 1.9.2 Fiji plugin (Liu et al. 2024, J Cell Sci 137:jcs261938; GPL-3.0; jar from
github.com/zs6e/excytosis-analyzer-plugin, sha256 b4826184…) on one stack from the command line.

ExoJ 1.9.2 has no macro mode, and the GitHub source is older than the jar (core classes differ), so
this driver does **not** re-implement anything. It opens the jar's own wizard, writes the values
into the wizard's `Parameters` object (what its text fields write into), presses Run/Next for
step 1 (spot detection) → 2 (tracking) → 3 (event identification), waits for each step, and saves
the wizard's own result table. Algorithm = the jar's, unchanged.

Install (once): put `ExoJ_1.9.2.jar` in `$FIJI/plugins`, then `FIJI=/path/to/Fiji.app ./build.sh` →
`$FIJI/plugins/ExoJ_Cli.jar`.

Run: `FIJI=/path/to/Fiji.app ./run_exoj.sh settings.properties [timeout_s]` (xvfb + `fiji -batch`;
`--headless` does not work, the wizard needs a display).

## Input

One single-channel 2D time series TIFF (T frames), 8/16/32-bit, drift-corrected beforehand
(ExoJ does not register). No saturated pixels (tutorial). Optional cell outline as ImageJ `.roi`.

## Settings (`key=value`; missing keys keep ExoJ's defaults)

| key | ExoJ field | meaning |
|---|---|---|
| `input`, `out_dir`, `name` | – | stack, output folder, file prefix |
| `pixel_size_um`, `frame_interval_s` | pixelSize, timeInterval | calibration (written onto the image) |
| `roi` | image ROI | detection region (.roi file) |
| `min_size`, `max_size` | minSize, maxSize | step 1 apparent spot size, **pixels (diameter-like), 1–25**, default 4–8; sets the wavelet scales |
| `snr_vesicle` | SNR_Vesicle | step 1 threshold k (× MAD), default 3 |
| `wavelet` | WaveletFilter | wavelet (true, default) or local maxima |
| `bleach_correct` | "Correct Photobleaching" | ExoJ's pixel-wise detrend |
| `delta_movie` | "dF movie" | detect on the first-order difference movie |
| `search_radius`, `temporal_depth`, `min_event_size` | step 2 | px, gap frames, min spots per track |
| `use_min_points`/`min_points` | step 3 | min fit points (5) |
| `use_expand`/`expand_pre`/`expand_post`/`fixed_expand` | step 3 | frames added before/after the track |
| `use_min_snr`/`min_snr` | step 3 | event threshold dF/σ(MAD) |
| `use_min_tau`/`min_tau`, `use_max_tau`/`max_tau` | step 3 | decay limits (frames) |
| `use_min_decay_r2`/`min_decay_r2` | step 3 | decay-fit R² |
| `use_min_size`/`min_est_size`, `use_max_size`/`max_est_size`, `use_min_size_r2`/`min_size_r2` | step 3 | apparent-size limits (turns the 2D Gaussian fit on) |
| `use_max_displacement`/`max_displacement` | step 3 | px |
| `timeout_s`, `exit` | – | per-step timeout; quit Fiji at the end (true) |

## Output (`out_dir`)

- `<name>_exoj_table.tsv` — the GUI's "ExoJ: Detected exocytosis list", cell for cell (τ in s).
- `<name>_events.tsv` — same events from the `Secretion` objects (τ in **frames**, F0, ΔF, SNR, peak x/y/frame…).
- `<name>_traces.tsv` — every point of every event trace (frame, x, y, ExoJ property, intensity).
- `<name>_params_used.tsv` — every `Parameters` field at each step + the snapshot step 3 used.
- `<name>_log.txt`.

## Checks (ExoJ 1.9.2, 2026-11)

Synthetic movie (256², 300 frames, 0.1 µm, 0.2 s; 30 events, Gaussian σ 1.5 px, τ 3–15 frames,
Poisson + Gaussian noise; `examples/synthetic.properties`, sizes 3–7): 30/30 found, 0 false,
onset within 0–2 frames (mean 0.13), τ ExoJ/true median 0.90 (r = 0.96); 2.2 s.
Repeat run: ExoJ table byte-identical; events.tsv fit values differ at ~1e-15 (parallel fitting).
