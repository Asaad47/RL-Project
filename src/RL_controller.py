import numpy as np
import random
import pystk
from PIL import Image
import matplotlib.pyplot as plt
from planner import load_model
import torchvision.transforms.functional as TF


# Constants
RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15
GAMMA = 0.99
LEARNING_RATE = 0.1  # Alpha for Q-learning
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 50000

# Action space
STEER_VALUES = [0, -1, 1]  # 3 values
ACCEL_VALUES = [1.0, 0.5, 0.05]  # 3 values
BRAKE_VALUES = [False, True]  # 2 values
DRIFT_VALUES = [False, True]  # 2 values
NITRO_VALUES = [True, False]  # 2 values

# Total number of actions: 3 * 3 * 2 * 2 * 2 = 72
# NUM_ACTIONS = len(STEER_VALUES) * len(ACCEL_VALUES) * len(BRAKE_VALUES) * len(DRIFT_VALUES) * len(NITRO_VALUES)
NUM_ACTIONS = len(STEER_VALUES)

# State discretization
IMG_WIDTH = 128
IMG_HEIGHT = 96
VELOCITY_BINS = 30  # Number of bins for velocity

MAX_SAME_POSITION_COUNT = 80

def get_action_from_index(action_idx):
    """Convert a single action index to the corresponding action tuple"""
    # nitro_idx = action_idx % len(NITRO_VALUES)
    # action_idx = action_idx // len(NITRO_VALUES)
    
    # drift_idx = action_idx % len(DRIFT_VALUES)
    # action_idx = action_idx // len(DRIFT_VALUES)
    
    # brake_idx = action_idx % len(BRAKE_VALUES)
    # action_idx = action_idx // len(BRAKE_VALUES)
    
    # accel_idx = action_idx % len(ACCEL_VALUES)
    # action_idx = action_idx // len(ACCEL_VALUES)
    
    # steer_idx = action_idx % len(STEER_VALUES)
    
    # return {
    #     'steer': STEER_VALUES[steer_idx],
    #     'acceleration': ACCEL_VALUES[accel_idx],
    #     'brake': BRAKE_VALUES[brake_idx],
    #     'drift': DRIFT_VALUES[drift_idx],
    #     'nitro': NITRO_VALUES[nitro_idx]
    # }
    
    steer_idx = action_idx % len(STEER_VALUES)
    
    return {
        'steer': STEER_VALUES[steer_idx],
        'acceleration': 1,
        'brake': False,
        'drift': False,
        'nitro': False
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

    def __init__(self, screen_width=128, screen_height=96):
        assert PyTux._singleton is None, "Cannot create more than one pytux object"
        PyTux._singleton = self
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
        
        velocity = np.linalg.norm(kart.velocity)
        velocity_bin = min(int(velocity), VELOCITY_BINS - 1)  # cap at VELOCITY_BINS
        # distance_down_track = kart.distance_down_track / track.length * 100
        # prev_distance_down_track = self.prev_distance_down_track / track.length * 100
        
        # return (*aim_point_image, int(velocity), int(distance_down_track), int(prev_distance_down_track))
        return (x,)


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
        
        # return np.array(self.k.render_data[0].image), None
        return state, track
    
    def get_reward(self, discrete_state, action):
        
        x = discrete_state[0]
        # velocity = discrete_state[1]
        
        reward = 0
        
        if action.steer != -1 and x < IMG_WIDTH / 2:
            reward -= (IMG_WIDTH - 2 * x) / IMG_WIDTH
        elif action.steer != 1 and x > IMG_WIDTH / 2:
            reward -= (2 * x - IMG_WIDTH) / IMG_WIDTH

        if action.rescue:
            reward -= 1
        # elif velocity < 5:
        #     reward -= 0.5
        # else:
        #     reward += 1
        
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
        
        self.prev_distance_down_track = kart.distance_down_track
        
        # print(f"t: {t}, kart.distance_down_track: {kart.distance_down_track}, current_vel: {current_vel}")
        
        # Check if kart needs rescue
        if t - last_rescue > RESCUE_TIMEOUT and current_vel < 1.0:
            last_rescue = t
            action.rescue = True
            # print(f">>> rescued. current_vel: {current_vel}")

        done = np.isclose(kart.overall_distance / track.length, 1.0, atol=2e-3)
        
        if done:
            print(f">>> finished, t: {t}")
            reward += 10
        
        reward += kart.distance_down_track / track.length
        # return np.array(self.k.render_data[0].image), reward, done, last_rescue
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
class Agent:
    def __init__(self, q_table_path=None):
        # Initialize Q-table with zeros
        # Shape: (img_width, img_height, velocity_bins, num_actions)
        self.q_table = np.zeros((IMG_WIDTH, NUM_ACTIONS))
        self.frame_idx = 0
        
        if q_table_path is not None:
            self.load_q_table(q_table_path)
        
    def load_q_table(self, path):
        self.q_table = np.load(path)

    def get_epsilon(self):
        return max(EPSILON_END + (EPSILON_START - EPSILON_END) * (1 - self.frame_idx / 1000), 0.1)

    def act(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.get_epsilon()
        if random.random() < epsilon:
            return random.randrange(NUM_ACTIONS)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        self.frame_idx += 1
        
        # Q-learning update
        current_q = self.q_table[state][action]
        next_q = np.max(self.q_table[next_state]) if not done else 0
        target_q = reward + GAMMA * next_q
        
        # Update Q-value
        self.q_table[state][action] += LEARNING_RATE * (target_q - current_q)
        
        return abs(target_q - current_q)  # Return TD error as loss

def run(track="zengarden", num_episodes=100, max_frames=1000, verbose=True, mode='train', q_table_path=None):
    env = PyTux()
    agent = Agent(q_table_path)
    # planner = load_model().eval()
    
    if mode == 'train':
        epsilon = None
    else:
        epsilon = 0.0 # no exploration in test mode
    
    if verbose:
        plt.ion()
        fig, ax = plt.subplots(1, 1)
    
    for episode in range(num_episodes):
        state, track_obj = env.reset(track)
        discrete_state = env.get_discrete_state(state, track_obj)
        episode_reward = 0
        last_rescue = 0
        cumulative_loss = 0
        
        for step in range(max_frames):
            
            action_idx = agent.act(discrete_state, epsilon)
            state, track_obj, reward, done, last_rescue = env.step(action_idx, step, last_rescue, discrete_state)
            next_discrete_state = env.get_discrete_state(state, track_obj)
            
            if env.same_position_count > MAX_SAME_POSITION_COUNT / 2:
                reward -= 2
            
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
                
                print(f"discrete_state: {discrete_state}, action_idx: {action_idx}")
                plt.pause(1e-3)
            
            if env.same_position_count > MAX_SAME_POSITION_COUNT:
                done = True
                print(f">>> same position for {env.same_position_count} frames. Killed after {step + 1} steps")
                env.prev_distance_down_track = 0
                env.same_position_count = 0
            
            if done:
                break
        
        if mode == 'train':
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Loss = {cumulative_loss:.4f}")
        else:
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        # Save Q-table periodically
        if (episode + 1) % 10 == 0:
            np.save(f'trained_models/qtable_ep{episode+1}.npy', agent.q_table)
            print(f"Q-table saved at episode {episode + 1}")
    
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
    parser.add_argument('--q_table_path', type=str, default=None,
                      help='Path to the Q-table file')
    
    args = parser.parse_args()
    run(args.track, args.episodes, args.max_frames, args.verbose, args.mode, args.q_table_path)
