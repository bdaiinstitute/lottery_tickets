# Generate demos using state-based planner

`mg_frankasim.py` works for all variants of the SQUIRL FrankaSim env, such as `PandaPickCube-v0` and `PandaPickCubeVision-v0`.

It's configured using hydra, see `cfgs`. Demos get saved into hydra output directories.

Datasets:

- demos_1k_PandaPickCube-v0.pkl, action_mag=\[0.004, 0.004\]: `gs://bdai-common-storage/squirl/frankasim/demos_1k_PandaPickCube-v0.pkl`
- demos_1k_PandaPickCubeRealisticControl-v0.pkl, action_mag=\[0.004, 0.004\]: `gs://bdai-common-storage/squirl/frankasim/demos_1k_PandaPickCubeRealisticControl-v0.pkl`
- demos_1k_PandaPickCube-v0.pkl, action_mag=\[0.002, 0.008\]: `gs://bdai-common-storage/squirl/frankasim/demos_1k_am_PandaPickCube-v0.pkl`
- demos_1k_PandaPickCubeRealisticControl-v0.pkl, action_mag=\[0.002, 0.008\]: `gs://bdai-common-storage/squirl/frankasim/demos_1k_am_PandaPickCubeRealisticControl-v0.pkl`
