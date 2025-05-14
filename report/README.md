# Scripts for testing

## Simple RL Controller

```bash
python simple_RL_controller.py --track zengarden --mode test --q_table_path trained_models/zengarden/simple_qtable_ep2910.npy --verbose
python simple_RL_controller.py --track lighthouse --mode test --q_table_path trained_models/lighthouse/simple_qtable_ep2910.npy --verbose
python simple_RL_controller.py --track hacienda --mode test --q_table_path trained_models/hacienda/simple_qtable_ep2970.npy --verbose
python simple_RL_controller.py --track snowtuxpeak --mode test --q_table_path trained_models/snowtuxpeak/simple_qtable_ep2920.npy --verbose
python simple_RL_controller.py --track cornfield_crossing --mode test --q_table_path trained_models/cornfield_crossing/simple_qtable_ep2980.npy --verbose
python simple_RL_controller.py --track scotland --mode test --q_table_path trained_models/scotland/simple_qtable_ep2910.npy --verbose
```

## Discrete RL Controller

```bash
python discrete_RL_controller.py --track zengarden --mode test --q_table_path trained_models/zengarden/discrete_qtable_ep2970.npy --verbose
python discrete_RL_controller.py --track lighthouse --mode test --q_table_path trained_models/lighthouse/discrete_qtable_ep2950.npy --verbose
python discrete_RL_controller.py --track hacienda --mode test --q_table_path trained_models/hacienda/discrete_qtable_ep2910.npy --verbose
python discrete_RL_controller.py --track snowtuxpeak --mode test --q_table_path trained_models/snowtuxpeak/discrete_qtable_ep2920.npy --verbose
python discrete_RL_controller.py --track cornfield_crossing --mode test --q_table_path trained_models/cornfield_crossing/discrete_qtable_ep2990.npy --verbose
python discrete_RL_controller.py --track scotland --mode test --q_table_path trained_models/scotland/discrete_qtable_ep2910.npy --verbose
```

## PPO

## DQN