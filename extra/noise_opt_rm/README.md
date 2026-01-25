# Robomimic (state-dim)
## How-to
```
# source dsrl_env
cd dsrl
```
### Searching for LTs  
Different methods to search for lottery tickets that depend on environment evaluation.
#### PRS: Pure random search  
Randomly sample tickets (noises from a Gaussian) and evaluate them over a fixed set of environments.
```
python noise_opt_rm/ticket_search/rollout/lottery_ticket.py \
--task_name can \
--n_envs 100 \
--noise_samples 5000 \
--seed 999 \
--out "logs_res_rm/lottery_ticket_results/" \
--ddim_steps 8 \
--no_wandb
```

#### ZOS: Zero Order Search  
Similar in implementation to [this](https://inference-scale-diffusion.github.io/) paper, but we additionally add a candidate along the winning direction of the previous iteration.
```
python noise_opt_rm/ticket_search/rollout/zo_opt.py \
--task_name can \
--n_envs 10 \
--n_iterations 100 \
--n_candidates 10 \
--lambda_radius 0.5 \
--seed 666 \
--n_seeds 10 \
--out "logs_res_rm/zo_ticket_results/" \
--ddim_steps 8 \
```

### Searching for LTs with Fixed Budgets
#### PRS
Randomly sample tickets (noises from a Gaussian) and evaluate them over a fixed set of environments. Also does PRS as `lottery_ticket.py`, but makes
searching with a given budget and generating results across multiple rng seeds easier (Splits `n_envs` into `n_seeds` and runs them in parallel, saving results for all the seeds every 100th noise evaluated.).
```
python noise_opt_rm/ticket_search/rollout/budget/lt.py \
--task_name can \
--n_envs 100 \
--noise_samples 1000 \
--seed 999 \
--n_seeds 10 \
--out "logs_res_rm/lottery_ticket_results/" \
--ddim_steps 8 \
--no_wandb
```



### Evaluations
#### Base Policy Eval  
Evaluate the base diffusion policies reslease by [DSRL](https://github.com/ajwagen/dsrl) (trained using [DPPO](https://github.com/irom-princeton/dppo)). The base policies can be found in this [drive link](https://drive.google.com/drive/folders/1kzC49RRFOE7aTnJh_7OvJ1K5XaDmtuh1).
```
python noise_opt_rm/eval/dppo_base_eval.py \
--task_name can \
--n_evals_per_seed 100 \
--n_seeds 5 \
--seed 1619 \
--out "logs_res_rm/policy_eval/" \
--ddim_steps 8
```
![Base policy: lift](../media/bp_lift_fail.gif)
![Base policy: can](../media/bp_can_fail.gif)


#### LT Eval
Evaluate the lottery tickets found previously by providing a path to the result directory. The ticket indices to be evaluated (tickets are ranked as per search performance  and then saved in that dir) can also be specfied.
```
python noise_opt_rm/eval/opt_noise.py \
--eval <path_to_lt_res_dir> \
--eval_idx {0..50} \
--task_name can \
--n_evals_per_seed 100 \
--n_seeds 5 \
--seed 1619 \
--out "logs_res_rm/noise_eval_results/" \
--ddim_steps 8
```
![Base policy: lift](../media/lt_lift_succ.gif)
![Base policy: can](../media/lt_can_succ.gif)


#### Budegt LT/ZO Eval
Over-engineered evaluation counterparts to `budget/lt.py`, `zo_opt.py`. Simply pass the the outer directory path generated from budget lt/zo search to the file and it will evaluate results for all seeds/episodic checkpoints.
```
python noise_opt_rm/eval/budget_opt_noise.py \
--ticket_path <path_to_lt_res_dir> \
--checkpoint 499 \
--task_name can \
--n_evals_per_seed 100 \
--n_seeds 5 \
--seed 1619 \
--out "logs_res_rm/noise_eval_results/" \
--ddim_steps 8
```


## Results
- LTs found using PRS (and budget) can be found in [lottery_ticket_results](results/lottery_ticket_results/) and their evaluations in [noise_eval_results](results/noise_eval_results/).
- LTs found using ZO can be found in [zo_ticket_results](results/zo_ticket_results/) and their evaluations in [zo_eval_results](results/zo_eval_results/).

## Important Implementation Details
### Wrapper order DPPO + Robomimic
```
MujocoEnv -> RobotEnv -> ManipulatorEnv -> PickPlace (Robosuite) -> RobosuiteEnv (Robomimic) -> RobomimicLowdimWrapper (DPPO) -> ObservationWrapperRobomimic (DSRL, parent: gym.Env) -> 
ActionChunkWrapper (p: gym.Env) -> VecEnv (SB3) -> DiffusionPolicyEnvWrapper (p: VecEnvWrapper- SB3)
```

#### Seeding
- The whole Robosuite and Robomimic stack does not set or use seeds or generators upto `1.4.1` (currently used by dsrl)
- However, the env wrappers enable the flow of seed till `MujocoEnv` which sets a `self.rng` in its init in `1.5.1` (used by vpl)- which is then used for sampling initial states
- `ActionChunkWrapper` passes up the seed in reset to `ObservationWrapperRobomimic`, which does seed `np.random`
- SB3 `VecEnv` defines a `_seed` which stores a list fo seeds for each env. This is initialized to None, and can be set using `seed()`- to ensure heterogeneous envs
- reset in SB3 `SubprocVecEnv` sends the respective seed to each env's FIRST reset call. SB3 `DummyVecEnv` does the same.

#### LT search v/s eval
- LT search needs to evaluate all the tickets on the same set of environments. For `noise_opt_rm`, the seeding for LT search is done through a backdoor created in `ActionChunkWrapper`, that sets the same seed again at every reset. An alternative way to acheve the same effect would have been to call SB3 vec_env.seed() at every reset, and then have SB3 reset pipe the env_seed to the repective envs. 
- For eval the seeding is done initially through the SB3 pipeline that sets the seed at the first reset only. It was noticed that parallel evaluations (using SB3 `SubprocVecEnv`) led to non-determinism, so the evaluation is restricted to singel environments at this time. 
- It is important to note that SB3 `SubprocVecEnv` internally resets the environment at every `done=True` so that episode collection is easier. The result is that the observation at every `done=True` is that of the next episode. Eval scripts in `noise_opt_rm` get around this by creating a new environment for every iteration. Lottery ticket search uses the seed backdoor such that the environments get reseeded with the same seed on every reset.

#### Success/rewards and done
- `is_success` is not used/set by DSRL, and success is inferred from reward thresholds for every step. 
- Technically DSRL does not exactly use a sparse reward for chunked actions, cause for every action within a chunk, they add a penalty of -1, and a +1 if the task is successful in that step. So for what is considered as a single environment step by DSRL, for a chunk of size 4, your reward could be among -4, -3, -2, -1, 0. DSRL logs a reward of 0 as a “success” for their policy, but SAC does get a bit denser reward.
- Original DSRL code did not use the truncation signal and always sets `done=true` after a fixed horizon in `ActionChunkWrapper`, irrespective of success or failure (that is, all the episodes ran to a fixed length). We observe much better performance for DSRL-SAC on setting the right termination and truncation signal which is used by SB3 downstream. 
- Rewards for LT search are calculated the same was as we would for SB3- stop counting once done is True. However, we give the reward offset of -1 a lower value while ranking LTs so that we don't get tickets that get ahead due to reward hacking (success is primary, but we still want to break ties using speed).
