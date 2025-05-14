#! /bin/bash

tracks=("zengarden" "lighthouse" "hacienda" "snowtuxpeak" "cornfield_crossing" "scotland")

for track in "${tracks[@]}"; do
    for i in {291..300}; do
        echo "Testing $track with episode $i"
        python simple_RL_controller.py --track $track --mode test --q_table_path trained_models/$track/simple_qtable_ep${i}0.npy
    done
done

for track in "${tracks[@]}"; do
    for i in {291..300}; do
        echo "Testing $track with episode $i"
        python discrete_RL_controller.py --track $track --mode test --q_table_path trained_models/$track/discrete_qtable_ep${i}0.npy
    done
done

