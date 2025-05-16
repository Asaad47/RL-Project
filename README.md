# Reinforcement Learning Project

## Installation

On M1 Mac machines:

Open the terminal using `Rosetta` and run the following commands:

```bash
CONDA_SUBDIR=osx-64 conda create -n tuxcart python=3.9
conda activate tuxcart
conda config --env --set subdir osx-64
pip install -U PySuperTuxKart
```

Install `numpy`, `matplotlib`, and `torch`.

## Commands for running the project

Inside the `src` directory, `simple_RL_controller.py`, `discrete_RL_controller.py`, `DQN.py`, and `neural_RL_controller.py` are the main scripts for running the project. They can be run with the following commands:

```bash
python simple_RL_controller.py [--track TRACK_NAME] [--episodes NUM_EPISODES] [--max_frames MAX_FRAMES] [--mode {test, train}] [--q_table_path Q_TABLE_PATH] [--verbose]
python discrete_RL_controller.py [--track TRACK_NAME] [--episodes NUM_EPISODES] [--max_frames MAX_FRAMES] [--mode {test, train}] [--q_table_path Q_TABLE_PATH] [--verbose]
python DQN.py [--track TRACK_NAME] [--episodes NUM_EPISODES] [--max_frames MAX_FRAMES] [--mode {test, train}] [--model_path MODEL_PATH] [--frame_skip FRAME_SKIP] [--verbose]
python neural_RL_controller.py [--track TRACK_NAME] [--episodes NUM_EPISODES] [--max_frames MAX_FRAMES] [--mode {test, train}] [--q_network_path Q_NETWORK_PATH] [--verbose]
```

All arguments are optional. If not provided, the default values will be used. The common default values are:

- `TRACK_NAME`: `zengarden`
- `NUM_EPISODES`: `1`
- `MAX_FRAMES`: `1000`
- `MODE`: `train`

To get the animation of the kart, run the scripts with `--verbose` flag.

Example training commands:

```bash
python simple_RL_controller.py --track zengarden --mode train --episodes 3000 --max_frames 2000 ;
python discrete_RL_controller.py --track zengarden --mode train --episodes 3000 --max_frames 2000 ;
python DQN.py --track zengarden --mode train --episodes 5000 --max_frames 2000 ;
python neural_RL_controller.py --track zengarden --mode train --episodes 5000 --max_frames 2000 ;
```

## Scripts for testing

### Simple RL Controller

```bash
python simple_RL_controller.py --track zengarden --mode test --q_table_path trained_models/zengarden/simple_qtable_ep2910.npy --verbose ;
python simple_RL_controller.py --track lighthouse --mode test --q_table_path trained_models/lighthouse/simple_qtable_ep2980.npy --verbose ;
python simple_RL_controller.py --track hacienda --mode test --q_table_path trained_models/hacienda/simple_qtable_ep2940.npy --verbose ;
python simple_RL_controller.py --track snowtuxpeak --mode test --q_table_path trained_models/snowtuxpeak/simple_qtable_ep2920.npy --verbose ;
python simple_RL_controller.py --track cornfield_crossing --mode test --q_table_path trained_models/cornfield_crossing/simple_qtable_ep3000.npy --verbose ;
python simple_RL_controller.py --track scotland --mode test --q_table_path trained_models/scotland/simple_qtable_ep2910.npy --verbose ;
```

Results:
- zengarden: 477 steps
- lighthouse: 501 steps
- hacienda: 550 steps
- snowtuxpeak: 611 steps
- cornfield_crossing: 693 steps
- scotland: 660 steps

### Discrete RL Controller

```bash
python discrete_RL_controller.py --track zengarden --mode test --q_table_path trained_models/zengarden/discrete_qtable_ep3000.npy --verbose ;
python discrete_RL_controller.py --track lighthouse --mode test --q_table_path trained_models/lighthouse/discrete_qtable_ep3000.npy --verbose ;
python discrete_RL_controller.py --track hacienda --mode test --q_table_path trained_models/hacienda/discrete_qtable_ep3000.npy --verbose ;
python discrete_RL_controller.py --track snowtuxpeak --mode test --q_table_path trained_models/snowtuxpeak/discrete_qtable_ep2930.npy --verbose ;
python discrete_RL_controller.py --track cornfield_crossing --mode test --q_table_path trained_models/cornfield_crossing/discrete_qtable_ep2970.npy --verbose ;
python discrete_RL_controller.py --track scotland --mode test --q_table_path trained_models/scotland/discrete_qtable_ep3000.npy --verbose ;
```

Results:
- zengarden: 630 steps
- lighthouse: 577 steps
- hacienda: 779 steps (gets stuck)
- snowtuxpeak: 788 steps
- cornfield_crossing: 837 steps
- scotland: 730 steps

### DQN

```bash
python DQN.py --track zengarden --mode test --verbose ;
python DQN.py --track lighthouse --mode test --verbose ;
python DQN.py --track hacienda --mode test --verbose ;
```



### Controller test

To get the animation of the kart, run the script with `-v` flag.

```bash
python controller.py zengarden lighthouse hacienda snowtuxpeak cornfield_crossing scotland

>>> rescued. current_vel: 0.029272134130120666
>>> rescued. current_vel: 0.19547797470751374
>>> rescued. current_vel: 0.834031620465489
631 0.9986513616504714 zengarden
❌ Missed the track limit by 131 steps
--------------------
>>> rescued. current_vel: 0.20168671868270702
503 0.998880237665428 lighthouse
❌ Missed the track limit by 3 steps
--------------------
>>> rescued. current_vel: 0.8269845822548917
649 0.9998108039523975 hacienda
✅ Passed the track limit by 51 steps!
--------------------
>>> rescued. current_vel: 0.2800737442979676
>>> rescued. current_vel: 0.5347946552341052
>>> rescued. current_vel: 0.8340231073920961
632 1.0003337786379827 snowtuxpeak
✅ Passed the track limit by 68 steps!
--------------------
>>> rescued. current_vel: 0.4741605441843571
>>> rescued. current_vel: 0.6344187605745537
764 0.998409503379093 cornfield_crossing
✅ Passed the track limit by 36 steps!
--------------------
>>> rescued. current_vel: 0.7549736571863991
>>> rescued. current_vel: 0.26727862444314143
696 0.9982932460059366 scotland
✅ Passed the track limit by 104 steps!
--------------------
```
