# Franka-sim Lottery Ticket Examples 

## Setup

First, install the `uv` package manager using [these instructions](https://docs.astral.sh/uv/getting-started/installation/).

Next, clone the repo and go into it.

```bash
git clone https://github.com/rai-inst/lottery_tickets.git
cd lottery_tickets
```

Then, setup the 

```bash
# Setup uv venv and install franka-sim
uv sync --extra franka-sim
# source the venv
source .venv/bin/activate

# Go into the franka_sim_lt folder
cd src/lottery_tickets/franka_sim_lt/

# It helps to set `MUJOCO_GL` to use gpu rendering for faster performance:
export MUJOCO_GL=egl
```

The codebase supports the following:
1. [Flow matching policies already trained that you can evaluate](#evaluating-pretrained-flow-matching-policy)
2. [Generating a new lottery ticket for franka-sim](#generating-a-new-lottery-ticket-for-franka-sim)
3. [Evaluating an existing franka-sim lottery ticket](#evaluating-an-existing-franka-sim-lottery-ticket)
4. [Visualize ticket and original policy performance](#visualize-ticket-and-original-policy-performance)
5. [Generate data and train your own base policy](#generate-data-and-train-your-own-base-policy)

## Evaluating pretrained flow matching policy
First, go into `train_model` folder, and download a checkpoint (TODO: Make this accessible to public)
```
cd train_model
gsutil -m cp -r  "gs://bdai-common-storage/lottery_tickets/checkpoints" .
```

You can run an evaluation on that checkpoint by running `evaluate.py` and setting `evaluation.model_path` to your chosen checkpoint.
You can set `num_episodes` to determime how many block poses we evaluate the policy on.
All of the outputs will be saved in `hydra.run.dir`, or `outputs` by default.
For these examples, consider the ckpt `fm_seed_1001`, and we'll save all our experiments in `fm_seed_1001_example`, and name this one `original_policy` for making it easy to compare against tickets later:

```
python evaluate.py evaluation.model_path=checkpoints/fm_seed_1001/checkpoints/fm_policy_final.pt +original_policy=True evaluation.num_episodes=10 hydra.run.dir=outputs/fm_seed_1001_example/original_policy 
```

You will see the policy's total rewards for each episode get printed out. This policy typically has an average episode reward of 20-40, whereas success normally is >80. It on occasion succeeds, slowly, but most of the time it is not a good policy. 

## Generating a new lottery ticket for franka-sim
You can generate a new lottery ticket and evaluate it by setting `new_noise=True`. It'll run similarly to the previous script, except an initial noise will be chosen at the start, used for all episodes, and then saved as `init_x.pt` in the same folder as the videos folder. Run the script to grab a lottery ticket and see if you win!

```
python evaluate.py evaluation.model_path=checkpoints/fm_seed_1001/checkpoints/fm_policy_final.pt +new_noise=True hydra.run.dir=outputs/fm_seed_1001_example/example_ticket evaluation.num_episodes=10
```

Depending on the base policy and the ticket drawn, the performance can cover a wide range. If you'd like to generate a large number of tickets, you can run the following bash script. It will generate n tickets, and make a folder for each of them inside `output_dir`. Example here does 25 tickets with 10 environment states (250 episodes), but you can vary this based on your compute budget. (TODO: remove this script and add functionality to python script, requires changing save file dir for hydra). This will generate a folder that will contain subdirs, each subdir representing the results of a ticket:

```
./generate_tickets.sh \
  --n=25 \
  --model_path=checkpoints/fm_seed_1001/checkpoints/fm_policy_final.pt \
  --output_dir=outputs/fm_seed_1001_example \
  --num_episodes=10 \
  --new_noise=true
```

## Evaluating an existing franka-sim lottery ticket
You can evaluate the saved `init_x.pt` of a model by passing a path as an argument to the script via `noise_path` parameter. For example, you can download a golden ticket for `fm_seed_1001` checkpoint (and the other checkpoints) we've found via:

```
gsutil -m cp -r "gs://bdai-common-storage/lottery_tickets/golden_tickets" .
```

Now you can evaluate the golden tickets, for example:

```
python evaluate.py evaluation.model_path=checkpoints/fm_seed_1001/checkpoints/fm_policy_final.pt +noise_path=./golden_tickets/fm_seed_1001/init_x.pt hydra.run.dir=outputs/fm_seed_1001_example/golden_ticket evaluation.num_episodes=10
```

This golden ticket typically averages at least above 100, which is normally a success. It does still occassionaly fail, but it is much more reliable than the original policy. 

## Visualize ticket and original policy performance
You can visualize the results in a 2D scatter plot, where the x-axis represents the rewards/success rate for the first 50% episodes, and the y-axis is the rewards/success rate for the second 50% episodes. The more linear this is, the more predictable performance of a golden ticket on a set of environment states generalize to others. To help with checking for this linearity, our graph includes a best-fit line along with r^2 value. Also, if the results of the original policy are also in the folder (and named `original_policy`), then it will be added to the plot with specialized coloring to compare tickets and original policy performance.

The script also prints out the tickets in order of their average episode rewards and task success rate, along with the original policy's at the top.

```
python viz_regression_to_mean.py --root_dir=./outputs/fm_seed_1001_example --out_avg=fm_seed_1001_example_rewards.png --out_success=fm_seed_1001_example_success.png --threshold=100
```

Here is an example output we got when running the script on `fm_seed_1001_example`. As can be seen by the line of best fit and the r^2 value, the performance of the tickets on the first set of episodes is highly predictive of its performance on the other episodes. Also, we can see that while the base policy (red) has low performance, there are many tickets (blue) that are much better, dramatically increasing base policy performance from ~15% to high ~90% success rate.

<table align="center">
  <tr>
    <td><img src="../../../media/fm_seed_1001_success_frankasim.png" width="360"/></td>
    <td><img src="../../../media/fm_seed_1001_rewards_frankasim.png" width="360"/></td>
  </tr>
  <tr>
    <td align="center"><em>Success rates for tickets and base policy (`fm_seed_1001`)</em></td>
    <td align="center"><em>Episode rewards for tickets and base policy (`fm_seed_1001`)</em></td>
  </tr>
</table>

## Generate data and train your own base policy
You can generate demonstration data and train your own policy, in case you'd like to experiment with different parts of the pipeline to investigate what causes golden tickets to occur.

First, you can generate expert data by running the following script:

```
cd generate_data
python generate_data.py
```

This will use a task and motion planning algorithm to generate demonstrations of the franka picking up the cube. By default, the script will run until it has collected 1000 succesful demos. All of the saved data will be placed in the `outputs` folder by default. There will be a pickle file `demos.pkl` that contains all the demonstrations and will be used for training.

You can also download the data we used to train our checkpoints here if you'd prefer not to generate your own data:

```
gsutil -m cp -r "gs://bdai-common-storage/lottery_tickets/data" .
```


Now we can train a policy with the data by using the train script inside `train_model`, and passing the path to `demo.pkl` to the `dataset.data_path` parameter. For example:

```
cd train_model
python train.py dataset.data_path=/PATH/TO/demos.pkl
```

This will print out the average loss per epoch, and save checkpoints along the way to `outputs/policy`. The last checkpoint will be saved as `fm_policy_final.pt`, although you can use any of the checkpoints saved along the way.

You can [evaluate your newly trained checkpoint](#evaluating-pretrained-flow-matching-policy) in the same way as before, or you can [search for golden tickets](#generating-a-new-lottery-ticket-for-franka-sim).

## Generate demos using state-based planner

`mg_frankasim.py` works for all variants of the SQUIRL FrankaSim env, such as `PandaPickCube-v0` and `PandaPickCubeVision-v0`.

It's configured using hydra, see `cfgs`. Demos get saved into hydra output directories.

Datasets:

- demos_1k_PandaPickCube-v0.pkl, action_mag=\[0.004, 0.004\]: `gs://bdai-common-storage/squirl/frankasim/demos_1k_PandaPickCube-v0.pkl`
- demos_1k_PandaPickCubeRealisticControl-v0.pkl, action_mag=\[0.004, 0.004\]: `gs://bdai-common-storage/squirl/frankasim/demos_1k_PandaPickCubeRealisticControl-v0.pkl`
- demos_1k_PandaPickCube-v0.pkl, action_mag=\[0.002, 0.008\]: `gs://bdai-common-storage/squirl/frankasim/demos_1k_am_PandaPickCube-v0.pkl`
- demos_1k_PandaPickCubeRealisticControl-v0.pkl, action_mag=\[0.002, 0.008\]: `gs://bdai-common-storage/squirl/frankasim/demos_1k_am_PandaPickCubeRealisticControl-v0.pkl`
