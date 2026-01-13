# DPPO for robomimic Lottery Ticket Examples

The original DPPO paper released state-based diffusion policy checkpoints for robomimic tasks.
These checkpoints were used in the original DSRL set of experiments.
We use these same model checkpoints and show the existence of golden tickets.

We have scripts for:
1. [Generating new tickets using a base policy](#generate-tickets-with-dppo-robomimic)
2. [Evaluate the default dppo robomic policy](#evaluate-the-default-dppo-robomic-policy)
3. [Evaluate golden tickets for dppo robomimic](#evaluate-golden-tickets-for-dppo-robomimic)

## Setup

Clone the repo **with submodules** and go into it.

```bash
git clone --recursive https://github.com/rai-inst/lottery_tickets.git
cd lottery_tickets
```

From the repo root, create a virtual environment with `uv`, and install the `dppo-robomimic` dependencies:

```bash
uv sync --extra dppo-robomimic
source .venv/bin/activate
```

**TODO: simplify the dependencies, not sure we need the stable-baselines for basic results?**


For robomimic, you will also need to install the MuJoCo 2.1 binaries as described in [this README](https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco).

**NOTE:** If you get an error about `GL/osmesa.h: No such file or directory` when running some of the examples below, run the following command ([source](https://github.com/ethz-asl/reinmav-gym/issues/35)):

```bash
sudo apt-get install libosmesa6-dev
```

## Download pretrained DPPO robomimic checkpoints

To download the pretrained DPPO model checkpoints, [download this folder](https://drive.google.com/drive/folders/1kzC49RRFOE7aTnJh_7OvJ1K5XaDmtuh1) and place it in your `dppo/log` folder.
This is directly taken from [the original DSRL codebase](https://github.com/ajwagen/dsrl?tab=readme-ov-file#installation).

## Generate tickets with DPPO robomimic

Randomly sample `noise_samples` tickets (noises from a Gaussian) and evaluate them over `n_envs` fixed set of environments.
Results are logged to `out`.
You can pass `--save-vid` to save videos for all tickets on all environments, but normally only recommended for n_envs < 10 for debugging purposes. 

```bash
python lottery_ticket.py \
    --task_name can \
    --n_envs 3 \
    --noise_samples 5 \
    --seed 999 \
    --out "logs_res_rm/lottery_ticket_results/" \
    --ddim_steps 8 \
    --no_wandb \
    --save_vid
```

## Evaluate the default DPPO robomimic policy

We can evaluate the base policy performance (i.e., sampling from Gaussian) with a similar script.
For the `can` task, we typically see a performance in the 40% success range for the base policy performance.

```bash
python dppo_base_eval.py \
    --task_name can \
    --n_evals_per_seed 3 \
    --n_seeds 50 \
    --seed 1619 \
    --out "logs_res_rm/policy_eval/" \
    --ddim_steps 8 
```

## Evaluate golden tickets for dppo robomimic

You can download a folder containing golden tickets for the `can` task [here](https://drive.google.com/drive/folders/1GCtMUE3bylCTIZb_zQYgCxVcrj_Phl3-).
You can then run the following script to evaluate the golden tickets on different environment states by passing the directory path to `eval` parameter.
The folder contains multiple tickets, ranked by their performance, so you can use `eval_idx` to select which ticket to run, with `0` representing the best golden ticket. 

Generally, the average success rate of the golden ticket in `envs100_samples5000_seed999_ddim8_20251130_221846_ddim8` for `eval_idx=0` is ~80% for the `can` task.

```bash
python opt_noise.py \
    --eval ./envs100_samples5000_seed999_ddim8_20251130_221846_ddim8 \
    --eval_idx 0 \
    --task_name can \
    --n_evals_per_seed 3 \
    --n_seeds 50 \
    --seed 1619 \
    --out "logs_res_rm/noise_eval_results/" \
    --ddim_steps 8 
```
