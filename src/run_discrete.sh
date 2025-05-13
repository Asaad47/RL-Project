#! /bin/bash

tracks=("zengarden" "lighthouse" "hacienda" "snowtuxpeak" "cornfield_crossing" "scotland")

for track in "${tracks[@]}"; do
    python discrete_RL_controller.py --track $track --mode train --max_frames 2000 --episodes 3000
done

for track in "${tracks[@]}"; do
    python discrete_RL_controller.py --track $track --mode test --q_table_path trained_models/$track/discrete_qtable_ep1000.npy
    python discrete_RL_controller.py --track $track --mode test --q_table_path trained_models/$track/discrete_qtable_ep2000.npy
    python discrete_RL_controller.py --track $track --mode test --q_table_path trained_models/$track/discrete_qtable_ep3000.npy
done