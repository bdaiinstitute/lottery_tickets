N=50
for ((i=1; i<=N; i++)); do
    python evaluate.py evaluation.model_path=checkpoints/fm_seed_1004/checkpoints/fm_policy_final.pt +new_noise=True hydra.run.dir=outputs/fm_seed_1004_lottery_ticket_search/$i evaluation.num_episodes=50
done