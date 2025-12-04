# Base policies trained with behavior cloning

Train a flow matching policy with `python train.py dataset.data_path=...`. Evaluate it with `python evaluate.py evaluation.model_path=...`. Configured using hydra, see `cfgs`.

Tested with machine generated demonstrations in FrankaSim, see `examples/demogen`.

Checkpoints:

- demos_1k_PandaPickCube-v0_checkpoints, action_mag=\[0.004, 0.004\]: `gs://bdai-common-storage/squirl/frankasim/demos_1k_PandaPickCube-v0_checkpoints`
- demos_1k_PandaPickCubeRealisticControl-v0_checkpoints, action_mag=\[0.004, 0.004\]: `gs://bdai-common-storage/squirl/frankasim/demos_1k_PandaPickCubeRealisticControl-v0_checkpoints`
- demos_1k_PandaPickCube-v0_checkpoints, action_mag=\[0.002, 0.008\]: `gs://bdai-common-storage/squirl/frankasim/demos_1k_am_PandaPickCube-v0_checkpoints`
- demos_1k_PandaPickCubeRealisticControl-v0_checkpoints, action_mag=\[0.002, 0.008\]: `gs://bdai-common-storage/squirl/frankasim/demos_1k_am_PandaPickCubeRealisticControl-v0_checkpoints`
