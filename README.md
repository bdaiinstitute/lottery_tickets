# The Lottery Ticket Hypothesis for Improving Pretrained Robot Diffusion and Flow Policies

This is a repository for testing the lottery ticket hypothesis for robot control. There are three different experimental setups, where each experiment uses a unique simulation and policy class:
1.  [LeRobot pretrained SmolVLA for LIBERO](#smolvla-for-libero-lottery-ticket-examples)
2. [State-based flow matching policy for franka-sim](#franka-sim-lottery-ticket-examples)
3. 🚧 State-based diffusion policies for robomimic 🚧

SmolVLA + Libero represents an experiment where a pretrained VLA checkpoint is taken (directly from LeRobot), and golden tickets are searched for over a multitude of task suites. We also include golden tickets we have found which can be evaluated.

Franka-sim involves a cube picking task with a franka robot, and includes an automated way to generate demonstrations, training code for behavior cloning with a flow matching policy on the collected data, and model checkpoints of policies we have already trained. We also include golden tickets for the checkpoints we provide.

🚧 State-based diffusion policies for robomimic 🚧


# SmolVLA for LIBERO Lottery Ticket Examples

For the pretrained policy weights in our experiments, we use <a href="https://huggingface.co/HuggingFaceVLA/smolvla_libero">LeRobot's finetuned version of SmolVLA for LIBERO: "HuggingFaceVLA/smolvla_libero" </a>. 

There are 5 libero environments you can use as your env.task:
- libero_object
- libero_spatial
- libero_goal
- libero_90
- libero_10

We have a python script (pretty much exact copy of `lerobot_eval.py`) you can run that can be used in 1 of 3 ways:
1. [Generate a new lottery ticket (i.e: get performance on a task suite)](#generating-a-new-ticket)
2. [Evaluate a saved lottery ticket on other tasks](#evaluating-a-saved-ticket)
3. [Running the original policy](#running-the-original-policy)


## Setup

```
# Clone the repo and go into it.
git clone https://github.com/rai-inst/lottery_tickets.git
cd lottery_tickets

# Make and activate conda environment.
conda create -n lottery_tickets python=3.10
conda activate lottery_tickets

# Install package with smolvla + libero dependencies
pip install -e .[smolvla-libero]

# It helps to set `MUJOCO_GL` to use gpu rendering for faster performance:
export MUJOCO_GL=egl

# Go into the `smolvla_libero` directory
cd src/lottery_tickets/smolvla_libero
```

## Generating a new ticket

Set `eval_mode=NEW_TICKET` to generate a new noise vector (it will be sampled from standard normal), and run `n_episodes` of eval on it for the `env.task` list. You can set the seed for the environments by passing `seed` parameter an integer argument (`1000` is the default value). The noise vector will be saved to `{output_dir}/{A_UNIQUE_ID}/initial_noise.pt` for future use, along with videos and results. 

```
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

Set `eval_mode=LOAD_TICKET` and load a ticket by passing `initial_noise.pt` into `noise_path`. We can change the seed too to rollout on different environment seeds.

```
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

Set `eval_mode=ORIGINAL_POLICY`, and the original policy (i.e: sampling from gaussian at all steps) will be evaluated. Results and videos will be saved, but there will be noise `initial_noise.pt` saved since it's not used. 

```
python evaluate.py \
        --policy.path="HuggingFaceVLA/smolvla_libero" \
        --env.type=libero \
        --env.task=libero_spatial \
        --eval.batch_size=1 \
        --eval.n_episodes=1 \
        --output_dir=outputs/libero_spatial_tickets \
        --eval_mode=ORIGINAL_POLICY \
        --seed=1000
```



# Franka-sim Lottery Ticket Examples 

## Setup
```
# Clone the repo and go into it.
git clone https://github.com/rai-inst/lottery_tickets.git
cd lottery_tickets

# Setup uv venv and install franka-sim
uv sync --extra franka-sim
# source the venv
source .venv/bin/activate

# Go into the  franka_sim_lt folder
cd src/lottery_tickets/franka_sim_lt/

# It helps to set `MUJOCO_GL` to use gpu rendering for faster performance:
export MUJOCO_GL=egl
```

The codebase supports the following:
1. [Flow matching policies already trained that you can evaluate](#evaluating-pretrained-flow-matching-policy)
2. TODO:

# Evaluating pretrained flow matching policy
First, go into `train_model` folder, and download a checkpoint (TODO: Make this accessible to public)
```
cd train_model
gsutil -m cp -r  "gs://bdai-common-storage/lottery_tickets/checkpoints"  .
```

You can run an evaluation on that checkpoint by running `evaluate.py` and setting `evaluation.model_path` to your chosen checkpoint.

```
python evaluate.py evaluation.model_path=checkpoints/fm_seed_1002/checkpoints/fm_policy_final.pt
```

You will see the policy's episode returns (typically above 100 means success), and a saved video of the rollout. 