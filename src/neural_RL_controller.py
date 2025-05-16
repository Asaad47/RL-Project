import numpy as np
import pandas as pd
import random
import pystk
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# Constants
RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15
GAMMA = 0.99
LEARNING_RATE = 0.1  # Alpha for Q-learning
EPSILON_START = 1.0
EPSILON_END = 0.1

# Action space
STEER_VALUES = [0, -1, 1]  # 3 values
ACCEL_VALUES = [0.8, 0.5, 0.05]  # 3 values
BRAKE_VALUES = [False, True]  # 2 values
DRIFT_VALUES = [False, True]  # 2 values
NITRO_VALUES = [True, False]  # 2 values

# Total number of actions: 3 * 3 * 2 * 2 * 2 = 72
NUM_ACTIONS = len(STEER_VALUES) * len(ACCEL_VALUES) * len(BRAKE_VALUES) * len(DRIFT_VALUES) * len(NITRO_VALUES)

# State discretization
IMG_WIDTH = 128
IMG_HEIGHT = 96

MAX_SAME_POSITION_COUNT = 60

def get_action_from_index(action_idx):
    """Convert a single action index to the corresponding action tuple"""
    nitro_idx = action_idx % len(NITRO_VALUES)
    action_idx = action_idx // len(NITRO_VALUES)
    
    drift_idx = action_idx % len(DRIFT_VALUES)
    action_idx = action_idx // len(DRIFT_VALUES)
    
    brake_idx = action_idx % len(BRAKE_VALUES)
    action_idx = action_idx // len(BRAKE_VALUES)
    
    accel_idx = action_idx % len(ACCEL_VALUES)
    action_idx = action_idx // len(ACCEL_VALUES)
    
    steer_idx = action_idx % len(STEER_VALUES)
    
    return {
        'steer': STEER_VALUES[steer_idx],
        'acceleration': ACCEL_VALUES[accel_idx],
        'brake': BRAKE_VALUES[brake_idx],
        'drift': DRIFT_VALUES[drift_idx],
        'nitro': NITRO_VALUES[nitro_idx]
    }


def create_action(action_dict):
    """Create a pystk.Action object from an action dictionary"""
    action = pystk.Action()
    action.steer = action_dict['steer']
    action.acceleration = action_dict['acceleration']
    action.brake = action_dict['brake']
    action.drift = action_dict['drift']
    action.nitro = action_dict['nitro']
    return action


# ===== Environment Wrapper =====
class PyTux:
    _singleton = None

    def __init__(self, mode, screen_width=128, screen_height=96):
        assert PyTux._singleton is None, "Cannot create more than one pytux object"
        PyTux._singleton = self
        self.mode = mode
        if mode == 'train':
            self.config = pystk.GraphicsConfig.ld()
        else:
            self.config = pystk.GraphicsConfig.hd()
        self.config.screen_width = screen_width
        self.config.screen_height = screen_height
        pystk.init(self.config)
        self.k = None
        
        self.prev_distance_down_track = 0
        self.same_position_count = 0

    @staticmethod
    def _point_on_track(distance, track, offset=0.0):
        """
        Get a point at `distance` down the `track`. Optionally applies an offset after the track segment if found.
        Returns a 3d coordinate
        """
        node_idx = np.searchsorted(track.path_distance[..., 1],
                                   distance % track.path_distance[-1, 1]) % len(track.path_nodes)
        d = track.path_distance[node_idx]
        x = track.path_nodes[node_idx]
        t = (distance + offset - d[0]) / (d[1] - d[0])
        return x[1] * t + x[0] * (1 - t)

    @staticmethod
    def _to_image(x, proj, view):
        p = proj @ view @ np.array(list(x) + [1])
        return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)
    
    def get_discrete_state(self, state, track):
        kart = state.players[0].kart
        
        proj = np.array(state.players[0].camera.projection).T
        view = np.array(state.players[0].camera.view).T
        aim_point_world = self._point_on_track(kart.distance_down_track+TRACK_OFFSET, track)
        aim_point_image = self._to_image(aim_point_world, proj, view)
        
        # Transform from [-1, 1] to pixel coordinates
        x = int((aim_point_image[0] + 1) / 2 * (IMG_WIDTH - 1))
        
        # Ensure coordinates are within image boundaries
        x = np.clip(x, 0, IMG_WIDTH - 1)
        
        percent_of_track = int(100 * kart.distance_down_track / track.length)
        percent_of_track = np.clip(percent_of_track, 0, 99)
        return (x, percent_of_track)


    def reset(self, track):
        if self.k is not None and self.k.config.track == track:
            self.k.restart()
            self.k.step()
        else:
            if self.k is not None:
                self.k.stop()
                del self.k
            config = pystk.RaceConfig(num_kart=1, laps=1, track=track)
            config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
            self.k = pystk.Race(config)
            self.k.start()
            self.k.step()
        
        state = pystk.WorldState()
        state.update()
        track = pystk.Track()
        track.update()
        
        return state, track
    
    def get_reward(self, discrete_state, action):
        
        x = discrete_state[0]
        percent_of_track = discrete_state[1]
        
        reward = 0
        
        if action.steer >= 0 and x < IMG_WIDTH / 2:
            reward -= (IMG_WIDTH - 2 * x) / IMG_WIDTH
        elif action.steer <= 0 and x > IMG_WIDTH / 2:
            reward -= (2 * x - IMG_WIDTH) / IMG_WIDTH
        else:
            reward += 0.01

        if action.rescue:
            reward -= 1
        else:
            reward += percent_of_track / 1000
        
        return reward

    def step(self, action, t, last_rescue, discrete_state):
        if isinstance(action, int) or isinstance(action, np.int64):
            # Convert action index to action dictionary
            action_dict = get_action_from_index(action)
            action = create_action(action_dict)
        elif isinstance(action, dict):
            # Convert action dictionary to pystk.Action
            action = create_action(action)
            
        # get reward from discrete state
        reward = self.get_reward(discrete_state, action)
        

        self.k.step(action)
        state = pystk.WorldState()
        state.update()
        track = pystk.Track()
        track.update()

        kart = state.players[0].kart
        current_vel = np.linalg.norm(kart.velocity)
        
        if abs(kart.distance_down_track - self.prev_distance_down_track) < 0.005:
            self.same_position_count += 1
        else:
            self.same_position_count = 0
            reward += 0.01
        
        self.prev_distance_down_track = kart.distance_down_track
        
        # print(f"t: {t}, kart.distance_down_track: {kart.distance_down_track}, current_vel: {current_vel}")
        
        # Check if kart needs rescue
        if current_vel < 1.0 and t - last_rescue > RESCUE_TIMEOUT:
            last_rescue = t
            action.rescue = True
            # print(f">>> rescued. current_vel: {current_vel}")

        done = np.isclose(kart.overall_distance / track.length, 1.0, atol=2e-3)
        
        if done:
            # print(f"!!>>> finished, t: {t}")
            reward += np.max([10 - (t / 100), 0.1])

        return state, track, reward, done, last_rescue

    def close(self):
        """
        Call this function, once you're done with PyTux
        """
        if self.k is not None:
            self.k.stop()
            del self.k
        pystk.clean()

# ===== Agent =====
class QNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=NUM_ACTIONS):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        state, action, reward, next_state, done = zip(*batch)
        return (torch.FloatTensor(state), 
                torch.LongTensor(action), 
                torch.FloatTensor(reward), 
                torch.FloatTensor(next_state), 
                torch.FloatTensor(done))
    
    def __len__(self):
        return len(self.buffer)

class NeuralAgent:
    def __init__(self, device=None, q_network_path=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        self.q_network = QNetwork().to(self.device)
        self.target_network = QNetwork().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer()
        self.frame_idx = 0
        self.batch_size = 64
        self.target_update = 10  # Update target network every 10 frames
        
        if q_network_path is not None:
            self.load_q_network(q_network_path)
    
    def load_q_network(self, path):
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_epsilon(self):
        return max(EPSILON_END + (EPSILON_START - EPSILON_END) * (1 - self.frame_idx / 1000), 0.1)
    
    def act(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.get_epsilon()
        
        if random.random() < epsilon:
            return random.randrange(NUM_ACTIONS)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def learn(self, state, action, reward, next_state, done):
        self.frame_idx += 1
        
        # Store transition in replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        # Only learn if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * GAMMA * next_q_values
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        if self.frame_idx % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()

def run(track="zengarden", num_episodes=100, max_frames=1000, verbose=True, mode='train', q_network_path=None):
    print(f"track: {track}, mode: {mode}")
    env = PyTux(mode)
    agent = NeuralAgent(q_network_path=q_network_path) if mode == 'train' else NeuralAgent(q_network_path=q_network_path)
    
    q_min = []
    q_max = []
    q_mean = []
    num_steps = []
    cumulative_rewards = []
    cumulative_losses = []
    finished_status = []
    
    if mode == 'train':
        epsilon = None
    else:
        epsilon = 0.0  # no exploration in test mode
    
    if verbose:
        plt.ion()
        fig, ax = plt.subplots(1, 1)
    
    for episode in range(num_episodes):
        state, track_obj = env.reset(track)
        discrete_state = env.get_discrete_state(state, track_obj)
        episode_reward = 0
        last_rescue = 0
        cumulative_loss = 0
        kill_episode = False
        
        for step in range(max_frames):
            action_idx = agent.act(discrete_state, epsilon)
            state, track_obj, reward, done, last_rescue = env.step(action_idx, step, last_rescue, discrete_state)
            next_discrete_state = env.get_discrete_state(state, track_obj)
            
            if mode == 'train':
                loss = agent.learn(discrete_state, action_idx, reward, next_discrete_state, done)
                cumulative_loss += loss
            discrete_state = next_discrete_state
            episode_reward += reward
            
            if verbose:
                ax.clear()
                ax.imshow(np.array(env.k.render_data[0].image))
                WH2 = np.array([env.config.screen_width, env.config.screen_height]) / 2
                
                kart = state.players[0].kart
                proj = np.array(state.players[0].camera.projection).T
                view = np.array(state.players[0].camera.view).T
                aim_point_world = env._point_on_track(kart.distance_down_track+TRACK_OFFSET, track_obj)
                aim_point_image = env._to_image(aim_point_world, proj, view)
                ax.add_artist(plt.Circle(WH2*(1+aim_point_image), 2, ec='r', fill=False, lw=1.5))   
                
                print(f"t: {step}, discrete_state: {discrete_state}, action_idx: {action_idx}, reward: {reward}, track percent: {int(100 * kart.distance_down_track / track_obj.length)}")
                plt.pause(1e-3)
            
            if not verbose and env.same_position_count > MAX_SAME_POSITION_COUNT:
                kill_episode = True
                env.prev_distance_down_track = 0
                env.same_position_count = 0
            
            if done or kill_episode:
                break
        
        if mode == 'train':
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Loss = {cumulative_loss:.4f}, distance_down_track: {state.players[0].kart.distance_down_track / track_obj.length:.2f}")
        else:
            print(f"Reward = {episode_reward:.2f}, steps: {step + 1}, distance_down_track: {state.players[0].kart.distance_down_track / track_obj.length:.2f}")
            print(f"overall_distance percent: {state.players[0].kart.overall_distance / track_obj.length:.2f}")
        
        # Save Q-network periodically
        if mode == 'train' and (episode + 1) % 10 == 0:
            torch.save(agent.q_network.state_dict(), f'trained_models/{track}/neural_qnetwork_ep{episode+1}.pt')
            q_min.append(np.min(agent.q_network(torch.FloatTensor([[0, 0]]).to(agent.device)).detach().cpu().numpy()))
            q_max.append(np.max(agent.q_network(torch.FloatTensor([[0, 0]]).to(agent.device)).detach().cpu().numpy()))
            q_mean.append(np.mean(agent.q_network(torch.FloatTensor([[0, 0]]).to(agent.device)).detach().cpu().numpy()))
            num_steps.append(step + 1)
            cumulative_rewards.append(episode_reward)
            finished_status.append(done)
            cumulative_losses.append(cumulative_loss)
    if mode == 'train':
        df = pd.DataFrame({
            'q_min': q_min,
            'q_max': q_max,
            'q_mean': q_mean,
            'num_steps': num_steps,
            'cumulative_rewards': cumulative_rewards,
            'cumulative_losses': cumulative_losses,
            'finished_status': finished_status
        })
        df.to_csv(f'trained_models/{track}/neural_qstats.csv', index=False)
    
    if verbose:
        plt.close()
    env.close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Q-learning agent with Q-table')
    parser.add_argument('--track', type=str, default='zengarden',
                      help='Track to train on')
    parser.add_argument('--episodes', type=int, default=1,
                      help='Number of episodes for training')
    parser.add_argument('--max_frames', type=int, default=1000,
                      help='Maximum number of frames per episode')
    parser.add_argument('--verbose', action='store_true',
                      help='Show verbose output during training')
    parser.add_argument('--mode', type=str, default='train',
                      help='Mode to run the script in')
    parser.add_argument('--q_network_path', type=str, default=None,
                      help='Path to the Q-network file')
    
    args = parser.parse_args()
    run(args.track, args.episodes, args.max_frames, args.verbose, args.mode, args.q_network_path)
