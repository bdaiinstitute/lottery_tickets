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
First, let's generate a bunch of expert data of the franka picking up a cube:

```
python src/lottery_tickets/franka_sim_lt/generate_data.py
```


# todos
- remove `mg_frankasim.py` when ready.
- get flow model
- get data collect
- train policy
- experiments for getting lottery tickets
- viz

- the `hil_gym` gym enviornment doesn't have a made where it returns both block position + images. For now, we just do block position, so written data has empty images + policy only operates on low-dim obs. We should support adding images for visuomotor policy testing.