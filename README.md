# 🎫 The Lottery Ticket Hypothesis for Improving Pretrained Robot Diffusion and Flow Policies 

  <div align="center">
    <h2>
      <a href="https://bdaiinstitute.github.io/lottery_tickets/">🔗 Project website</a>
    </h2>
  </div>
<table>
  <!-- Header row -->
  <tr>
    <th align="center">Original Policy (i.e: Initial noise sampled from gaussian)</th>
    <th align="center">🎫 Golden Ticket (i.e: Optimized fixed initial noise)</th>
  </tr>

  <!-- FrankaSim row -->
  <tr>
    <td align="center">
      <img src="./media/base_policy_frankasim.gif" width="360">
    </td>
    <td align="center">
      <img src="./media/golden_ticket_frankasim.gif" width="360">
    </td>
  </tr>

  <!-- LIBERO row -->
  <tr>
    <td align="center">
      <img src="./media/base_policy_libero.gif" width="360">
    </td>
    <td align="center">
      <img src="./media/golden_ticket_libero.gif" width="360">
    </td>
  </tr>

  <!-- Robomimic row -->
  <tr>
    <td align="center">
      <img src="./media/base_policy_robomimic.gif" width="360">
    </td>
    <td align="center">
      <img src="./media/golden_ticket_robomimic.gif" width="360">
    </td>
  </tr>


  <!-- Caption -->
  <tr>
    <td colspan="3" align="center">
      <em>
(left) Baseline policy (Gaussian sampling) vs. (right) 🎫 golden-ticket policy using a fixed initial noise. (top) is frankasim, (middle) is 🤗 SmolVLA + LIBERO, (bottom) is <a href="https://github.com/irom-princeton/dppo">DPPO + robomimic</a>. Each row uses a different golden ticket that was optimized for that model.
      </em>
    </td>
  </tr>
</table>


# Overview 

This is a repository for testing the lottery ticket hypothesis for robot control:
The performance of a pretrained, frozen diffusion/flow matching policy can be improved by replacing sampling initial noise from the prior distribution (typically isotropic gaussian) with a well-chosen, constant initial noise input, which we call a **golden ticket**.
There are three different experimental setups, where each experiment uses a unique simulation and policy class:

1. <a href="./src/lottery_tickets/franka_sim_lt/README.md">franka-sim cube picking with state-based flow matching policies</a>
2. <a href="./src/lottery_tickets/smolvla_libero/README.md">🤗 LeRobot pretrained 🤗SmolVLA for LIBERO</a>
3. <a href="./src/lottery_tickets/robomimic_dppo_lt/README.md">DPPO + robomimic</a>

All three experiment setups contain their own READMEs and code for running a baseline policy, generating tickets, evaluating tickets, and links to golden tickets we have found so you can try them yourself.
Each subfolder may contain other utilities, since each experiment testbed serves a different purpose:

<a href="./src/lottery_tickets/franka_sim_lt/README.md">🦾 Franka-sim</a> involves a cube picking task with a Franka robot, where the cube randomly spawns in a ~1/2 square meter region in front of the robot.
Our codebase includes an automated way to generate demonstrations, training code for behavior cloning with a flow matching policy on the collected data, and model checkpoints of policies we have already trained.
We also include golden tickets for the checkpoints we provide.
This is a great experimental testbed if you'd like to examine all parts of a pipeline (data collection, policy training, and inference) that result in policies with golden tickets.
The small model makes it easier to do experiments with little compute.
The policy and training code is all custom-written.

<a href="./src/lottery_tickets/smolvla_libero/README.md">🤗 SmolVLA + Libero</a> represents an experiment where a pretrained VLA checkpoint is taken (directly from LeRobot), and golden tickets are searched for over a multitude of task suites. 
We also include golden tickets we have found which can be evaluated.
This is a good experimental testbed for examining lottery tickets with an open-source VLA, and on a multi-task setting.
The policy used in our experiments comes from an off-the-shelf LIBERO checkpoint from LeRobot, so this reflects looking for lottery tickets in a model we didn't create. 

<a href="./src/lottery_tickets/robomimic_dppo_lt/README.md">✨ DPPO for robomimic</a> includes the original DPPO robomimic checkpoints used in the DSRL project.
We provide golden tickets for these policies, and code for generating new tickets and comparing against the base policy. 
This reflects a setting of using a model we didn't create.

# Getting started

We use some features of the [`uv`](https://docs.astral.sh/uv/) package manager in our `pyproject.toml`.
The easiest way to get started is to install `uv` using [these instructions](https://docs.astral.sh/uv/getting-started/installation/).

You can then install the individual experiment setups using `uv sync --extra $EXPERIMENT_NAME`; see the individual READMEs for more details.

# Contribution and Maintenance

This repository is released as-is to accompany a paper submission.

If you find any bugs, corrections, or issues that should be resolved for anyone looking to reproduce the results in this repository, please file an issue and we will look at it as soon as we can.

For other improvements, including new features, we recommend creating your own fork of the repository.
