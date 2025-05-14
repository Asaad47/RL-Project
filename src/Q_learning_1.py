import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pystk
import torchvision.transforms.functional as TF
from PIL import Image

# Constants
RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15

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
        
        return np.array(self.k.render_data[0].image), None

    def step(self, action, t, last_rescue):
        if isinstance(action, int):
            # Convert action index to action dictionary
            action_dict = get_action_from_index(action)
            action = create_action(action_dict)
        elif isinstance(action, dict):
            # Convert action dictionary to pystk.Action
            action = create_action(action)
            

        self.k.step(action)
        state = pystk.WorldState()
        track = pystk.Track()
        state.update()
        track.update()

        kart = state.players[0].kart
        current_vel = np.linalg.norm(kart.velocity)
        
        
        # Calculate reward based on progress and speed
        reward = 0
        if kart.distance_down_track - self.prev_distance_down_track < 0.01 or current_vel < 5.0:
            reward -= 0.5
        
        if action.rescue:
            reward -= 0.5
        else:
            reward += (kart.distance_down_track / track.length)
            
        # print(f"t: {t}, kart.distance_down_track: {kart.distance_down_track}, current_vel: {current_vel}")
        
        # Check if kart needs rescue
        if t - last_rescue > RESCUE_TIMEOUT and (current_vel < 1.0 or kart.distance_down_track - self.prev_distance_down_track < 0.005):
            last_rescue = t
            action.rescue = True
            # print(f">>> rescued. current_vel: {current_vel}")

        done = np.isclose(kart.overall_distance / track.length, 1.0, atol=2e-3)
        
        if done:
            reward += np.max([10 - (t / 100), 0.1])
            print(">>> arrived!")

        if kart.distance_down_track == self.prev_distance_down_track:
            self.same_position_count += 1
        else:
            self.same_position_count = 0
        self.prev_distance_down_track = kart.distance_down_track
        
        return np.array(self.k.render_data[0].image), reward, done, last_rescue

    def close(self):
        """
        Call this function, once you're done with PyTux
        """
        if self.k is not None:
            self.k.stop()
            del self.k
        pystk.clean()

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.0025
MEMORY_SIZE = 100_000
BATCH_SIZE = 32
TARGET_UPDATE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 50000
STATE_STACK = 4
IMG_SHAPE = (128, 96)  # Original image size
MAX_SAME_POSITION_COUNT = 60

# Action space
STEER_VALUES = [0, -1, 1]  # 3 values
ACCEL_VALUES = [1.0, 0.5, 0.05]  # 3 values
BRAKE_VALUES = [False, True]  # 2 values
DRIFT_VALUES = [False, True]  # 2 values
NITRO_VALUES = [True, False]  # 2 values

# Total number of actions: 3 * 3 * 2 * 2 * 2 = 72
NUM_ACTIONS = len(STEER_VALUES) * len(ACCEL_VALUES) * len(BRAKE_VALUES) * len(DRIFT_VALUES) * len(NITRO_VALUES)

def get_action_from_index(action_idx):
    """Convert a single action index to the corresponding action tuple"""
    # Calculate indices for each action component
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

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            # Input: [batch_size, STATE_STACK, 128, 96]
            nn.Conv2d(STATE_STACK, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # Output: [batch_size, 32, 31, 23]
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Output: [batch_size, 64, 14, 10]
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
            # Output: [batch_size, 64, 12, 8]
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 12 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def preprocess(img):
    # Convert numpy array to PIL Image
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    # Convert to grayscale
    img = TF.to_grayscale(img)
    
    # Convert to tensor and normalize
    img = TF.to_tensor(img)
    return img.squeeze(0).numpy()  # Convert back to numpy for storage

def get_epsilon(frame_idx):
    return max(EPSILON_END + (EPSILON_START - EPSILON_END) * (1 - frame_idx / 1000), 0.1)

def select_action(model, state, epsilon):
    if random.random() < epsilon:
        return random.randrange(NUM_ACTIONS)
    with torch.no_grad():
        # Reshape state to [1, STATE_STACK, height, width]
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        return model(state).argmax(1).item()

def optimize(model, target_model, memory, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    state, action, reward, next_state, done = memory.sample(BATCH_SIZE)

    # Reshape states to [batch_size, STATE_STACK, height, width]
    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.LongTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    done = torch.FloatTensor(done).to(device)

    q_values = model(state)
    next_q_values = target_model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + GAMMA * next_q_value * (1 - done)

    # Replace MSE loss with Huber loss (smooth L1 loss)
    # loss = nn.functional.smooth_l1_loss(q_value, expected_q_value)
    loss = nn.functional.mse_loss(q_value, expected_q_value)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def run(track="zengarden", device='cuda', num_episodes=100, max_frames=1000, verbose=True, mode='train', model_path=None, frame_skip=4):
    model = DQN(NUM_ACTIONS).to(device)
    if mode == 'train':
        target_model = DQN(NUM_ACTIONS).to(device)
        target_model.load_state_dict(model.state_dict())
        optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
        memory = ReplayBuffer(MEMORY_SIZE)
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

    env = PyTux()
    
    if mode != 'train' and verbose:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)

    # Initialize state stack
    state = preprocess(env.reset(args.track)[0])
    state_stack = deque([state] * STATE_STACK, maxlen=STATE_STACK)
    episode_reward = 0
    frame_idx = 0

    for episode in range(1, num_episodes + 1):
        obs = env.reset(track)[0]
        state = preprocess(obs)
        state_stack = deque([state] * STATE_STACK, maxlen=STATE_STACK)
        episode_reward = 0
        last_rescue = 0
        cumulative_loss = 0
        kill_episode = False
        
        for step in range(max_frames):
            frame_idx += 1
            stacked_state = np.array(state_stack)
            if mode == 'train':
                epsilon = get_epsilon(frame_idx)
            else:
                epsilon = 0
                
            # Only select new action every frame_skip frames
            if step % frame_skip == 0:
                action_idx = select_action(model, stacked_state, epsilon)
                action_dict = get_action_from_index(action_idx)
            
            # Apply the same action for frame_skip frames
            obs, reward, done, last_rescue = env.step(action_dict, step, last_rescue)
            next_state = preprocess(obs)
            state_stack.append(next_state)

            if mode == 'train':
                next_stacked = np.array(state_stack)

                # Only store transitions when we actually selected a new action
                if step % frame_skip == 0:
                    memory.push((stacked_state, action_idx, reward, next_stacked, done))
                    loss = optimize(model, target_model, memory, optimizer)
                    if loss is not None:
                        cumulative_loss += loss

                if frame_idx % TARGET_UPDATE == 0:
                    target_model.load_state_dict(model.state_dict())
            elif verbose:
                ax.clear()
                ax.imshow(obs)
                plt.pause(1e-3)

            episode_reward += reward
            
            if not verbose and env.same_position_count > MAX_SAME_POSITION_COUNT:
                kill_episode = True
                env.prev_distance_down_track = 0
                env.same_position_count = 0
            
            if done or kill_episode:
                break

        print(f"Episode {episode}: reward = {episode_reward}, cumulative loss = {cumulative_loss:.4f}, steps = {step + 1}, done = {done}")
        
        # Save model periodically
        if episode % 10 == 0:
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    if mode == 'train':
        # Save final model
        torch.save(model.state_dict(), model_path)
        print(f"Final model saved to {model_path}")
    elif verbose:
        plt.close()
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or test with DQN agent')
    parser.add_argument('--track', type=str, default='zengarden',
                      help='Track to train/test on')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                      help='Mode: train or test')
    parser.add_argument('--model_path', type=str, default='trained_models/dqn_model.pth',
                      help='Path to save/load the model')
    parser.add_argument('--episodes', type=int, default=1,
                      help='Number of episodes for training')
    parser.add_argument('--max_frames', type=int, default=1000,
                      help='Maximum number of frames to play')
    parser.add_argument('--verbose', action='store_true',
                      help='Show verbose output during playing')
    parser.add_argument('--frame_skip', type=int, default=4,
                      help='Number of frames to repeat each action')
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("device: ", device)

    run(track=args.track, 
        device=device, 
        num_episodes=args.episodes, 
        max_frames=args.max_frames, 
        verbose=args.verbose, 
        mode=args.mode, 
        model_path=args.model_path,
        frame_skip=args.frame_skip
    )