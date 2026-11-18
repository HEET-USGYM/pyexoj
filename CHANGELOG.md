# Changelog

## 0.1.0 (unreleased)

- Port of ExoJ 1.9.2 steps 1–3 (spot detection, tracking, event identification) and its result table.
- Opt-in exact speed-ups: `workers` (processes for steps 1 and 3), `fast` (in-place wavelet smoothing,
  no whole-frame copies per spot, grid search for link candidates), `device` (step 1 on a GPU via PyTorch).
- Tests against outputs of the original ExoJ 1.9.2 jar on a deterministic synthetic movie.
