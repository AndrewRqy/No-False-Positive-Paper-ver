# dataset_creation

Kubric ball-video stimulus generator and the velocity-profile designs used by the NFP test.

- `nfp_ball_dataset.py` -- the Kubric/Blender generator (`--profile_spec` renders a saved design).
- `design_decorrelated_stimulus.py` -- v2 design: LP that zeroes all 10 within-video tau couplings.
- `design_correlated_stimulus.py` -- v3 design: hold ONE pair at within-video correlation ~0.9
  (variant S = accel-speed, variant X = accel-vel_x). Adds a geometric-speed family to the generator.
- `specs/` -- rendered-set profile specs: `nfp_v2_profile_spec.json`, `nfp_v3_{S,X}_profile_spec.json`.
- `render/` -- Docker (Kubric) render scripts. Copy the chosen spec next to the generator or pass
  its full path. Rendered videos are hosted on HuggingFace (AndrewRqy/temporal-sae-videomae).

Design: `python -m dataset_creation.design_correlated_stimulus --N 3000`
Render: `data/render/run_nfp_v3_dataset.ps1 -Variant S` (Docker Desktop + kubricdockerhub/kubruntu).
