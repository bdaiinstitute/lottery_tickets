# SmolVLA for LIBERO Lottery Ticket Examples

For the pretrained policy weights in our experiments, we use <a href="https://huggingface.co/HuggingFaceVLA/smolvla_libero">LeRobot's finetuned version of SmolVLA for LIBERO: "HuggingFaceVLA/smolvla_libero" </a>. 

There are 5 libero environments you can use as your `env.task`:
- `libero_object`
- `libero_spatial`
- `libero_goal`
- `libero_90`
- `libero_10`

All experiment scripts run from the `smolvla_libero` folder, `src/lottery_tickets/smolvla_libero/`. We have a Python script (a lightly modified copy of `lerobot_eval.py`) that can be used to:
1. [Generate a new lottery ticket (i.e: get performance on a task suite)](#generating-a-new-ticket)
2. [Evaluate a saved lottery ticket on other tasks](#evaluating-a-saved-ticket)
3. [Running the original policy](#running-the-original-policy)
4. TODO: Visualize the results

## Setup
We include setup instructions for uv (which we recommend), and conda. Additionally, it helps to set `MUJOCO_GL` to use gpu rendering for faster performance:

```bash
export MUJOCO_GL=egl
```

### uv setup
From the repo root, create a virtual environment with `uv`, and install the `smolvla` and `libero` dependencies:

```bash
# Note: If git lfs causes issues, you can skip it with the following export
# export GIT_LFS_SKIP_SMUDGE=1
uv sync --extra smolvla-libero
source .venv/bin/activate
```

### conda setup
You can also setup with conda if you prefer:

```
conda create -n lottery_tickets python=3.10
conda activate lottery_tickets
pip install -e .[smolvla-libero]
```

## 🐛 Debugging SmolVLA + LIBERO setup
When installing the `smolvla-libero` dependencies, if you run into an issue with building `hf-egl-probe` and `egl-probe`, you may need to do:

```bash
uv pip install egl_probe --no-build-isolation
uv pip install hf_egl_probe --no-build-isolation
```


## Generating a new ticket

Set `eval_mode=NEW_TICKET` to generate a new noise vector (it will be sampled from standard normal), and run `n_episodes` of eval on it for the `env.task` list.
You can set the seed for the environments by passing `seed` parameter an integer argument (`1000` is the default value).
The noise vector will be saved to `{output_dir}/{A_UNIQUE_ID}/initial_noise.pt` for future use, along with videos and results.
(**TODO: batch_size for now is always assumed to be 1, but could be adjusted**).

You will be prompted to optionally specify a custom dataset folder path; the default is generally fine.

```bash
python evaluate.py \
        --policy.path="HuggingFaceVLA/smolvla_libero" \
        --env.type=libero \
        --env.task=libero_spatial \
        --eval.batch_size=1 \
        --eval.n_episodes=1 \
        --output_dir=outputs/libero_spatial_tickets \
        --eval_mode=NEW_TICKET \
        --seed=1000
```


## Evaluating a saved ticket

Set `eval_mode=LOAD_TICKET` and load a ticket by passing `initial_noise.pt` into `noise_path`.
You can change the `--seed` argument to rollout on different environment seeds.

We provide golden tickets for the different task suites in `lottery_tickets/src/lottery_tickets/smolvla_libero/golden_tickets`. Each folder contains folders that contain `initial_noise.pt` you can try.

```bash
python evaluate.py \
        --policy.path="HuggingFaceVLA/smolvla_libero" \
        --env.type=libero \
        --env.task=libero_spatial \
        --eval.batch_size=1 \
        --eval.n_episodes=1 \
        --output_dir=outputs/eval_libero_spatial_tickets/ticket_results \
        --eval_mode=LOAD_TICKET \
        --noise_path=PATH/TO/initial_noise.pt \
        --seed=100000
```


## Running the original policy

Set `eval_mode=ORIGINAL_POLICY`, and the original policy (i.e., sampling from gaussian at all steps) will be evaluated.
Results and videos will be saved to `{output_dir}/original_policy`, but there will be no `initial_noise.pt` saved since it's not used.
You can vary `n_episodes` to run the original policy multiple times on each task in the task suite `env.task`.

```bash
python evaluate.py \
        --policy.path="HuggingFaceVLA/smolvla_libero" \
        --env.type=libero \
        --env.task=libero_spatial \
        --eval.batch_size=1 \
        --eval.n_episodes=3 \
        --output_dir=outputs/libero_spatial_tickets \
        --eval_mode=ORIGINAL_POLICY \
        --seed=1000
```
