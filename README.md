# The Lottery Ticket Hypothesis for Improving Pretrained Robot Diffusion and Flow Policies

# Setup

```
# Clone the repo and go into it.
git clone https://github.com/rai-inst/lottery_tickets.git
cd lottery_tickets

# Make and activate conda environment.
conda create -n lottery_tickets python=3.10
conda activate lottery_tickets

# Install package with frankasim dependencies
pip install -e .[franka-sim]
```

# Franka-sim Lottery Ticket Examples
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

# todos
- example for running policy with gym env.
- experiments for getting lottery tickets.
- examples of pretrained policies and golden tickets we have found
- viz.
- speed stuff
- linting, etc.
- tests

- the `hil_gym` gym enviornment doesn't have a made where it returns both block position + images. For now, we just do block position, so written data has empty images + policy only operates on low-dim obs. We should support adding images for visuomotor policy testing.