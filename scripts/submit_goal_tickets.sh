#!/bin/bash
# Submit evaluation jobs for all goal tickets

TICKET_DIR="/home/opatil3/src/lottery_tickets/src/lottery_tickets/smolvla_libero/golden_tickets/libero_goal_tickets"

for ticket_folder in "$TICKET_DIR"/*/; do
    ticket_hash=$(basename "$ticket_folder")
    noise_path="$ticket_folder/initial_noise.pt"
    
    if [[ -f "$noise_path" ]]; then
        echo "Submitting job for ticket: $ticket_hash"
        sbatch -J "gl_$ticket_hash" lt_eval_goal.sh "$noise_path"
    else
        echo "Warning: No initial_noise.pt found in $ticket_folder"
    fi
done

echo "All jobs submitted!"
