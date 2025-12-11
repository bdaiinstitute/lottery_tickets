# The Lottery Ticket Hypothesis for Improving Pretrained Robot Diffusion and Flow Policies

# Setup

```
# Clone the repo and go into it.
git clone https://github.com/rai-inst/lottery_tickets.git
cd lottery_tickets

# Make and activate conda environment.
conda create -n lottery_tickets python=3.10
conda activate lottery_tickets

# Install package with smolvla + libero dependencies
pip install -e .[smolvla-libero]

# 🚧 (NOT STABLE) 🚧  Install package with frankasim
pip install -e .[franka-sim]
```

# SmolVLA for LIBERO Lottery Ticket Examples
First go into the `smolvla_libero` directory, and set `MUJOCO_GL` to use gpu rendering for faster performance:
```
cd src/lottery_tickets/smolvla_libero
export MUJOCO_GL=egl
```

For the pretrained policy weights in our experiments, we use <a href="https://huggingface.co/HuggingFaceVLA/smolvla_libero">LeRobot's finetuned version of SmolVLA for LIBERO: "HuggingFaceVLA/smolvla_libero" </a>. 

There are 5 libero environments you can use as your env.task:
- libero_object
- libero_spatial
- libero_goal
- libero_90
- libero_10

We have a python script (pretty much exact copy of `lerobot_eval.py`) you can run that can be used in 1 of 3 ways:

## 1. Generating a new ticket

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

## 2. Evaluating a saved ticket

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

## 3. Running the original policy

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



# 🚧 (NOT STABLE) 🚧 Franka-sim Lottery Ticket Examples 
First, generate a bunch of expert data of the franka picking up a cube:

```
cd src/lottery_tickets/franka_sim_lt/generate_data
python generate_data.py
```

This will save a `demo.pkl` file with all the succesful episodes, and the path to that file will get printed out at the end.

Next, train a simple flow matching model on the data.

```
cd src/lottery_tickets/franka_sim_lt/train_model
python train.py dataset.data_path=/PATH/TO/demos.pkl
```

This will train a flow model and output the paths to where all the checkpoints (including the final one `fm_policy.pt`) are saved. 

Now we can evaluate a model:

```
cd src/lottery_tickets/franka_sim_lt/train_model
python evaluate.py evaluation.model_path=/PATH/TO/fm_policy_final.pt
```

# todos
- example for running policy with gym env.
- experiments for getting lottery tickets.
- examples of pretrained policies and golden tickets we have found
- viz.
- speed stuff
- linting, etc.
- tests

- the `hil_gym` gym enviornment doesn't have a made where it returns both block position + images. For now, we just do block position, so written data has empty images + policy only operates on low-dim obs. We should support adding images for visuomotor policy testing.