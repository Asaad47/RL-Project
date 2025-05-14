import pystk
import numpy as np
import torchvision.transforms.functional as TF



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
    
    """
    aim_point[0]: in [0, ..., 127]
    aim_point[1]: in [0, ..., 95]
    current_vel: in [0, ..., 30]
    state: [aim_point[0], aim_point[1], current_vel]
    state possibilities: 128 * 96 * 30 = 368,640
    possible actions: 9 * 11 * 2 * 2 * 2 = 792
    Q-table size: 368,640 * 792 = 292,032,000
    """
    def Q_learning(state, action, reward, next_state, num_episodes=1000):
        """
        state: [aim_point[0], aim_point[1], current_vel]
        action: [steer, acceleration, brake, drift, nitro]
        reward: float
        next_state: [aim_point[0], aim_point[1], current_vel]
        """
        # Q-learning algorithm
        # Initialize Q-table
        # 5 actions: steer, acceleration, brake, drift, nitro
        # steer: [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
        # acceleration: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        # brake: [False, True]
        # drift: [False, True]
        # nitro: [False, True]
        Q = np.zeros((128, 96, 30, 9, 11, 2, 2, 2))  
        
        # Hyperparameters
        alpha = 0.1  # Learning rate
        gamma = 0.9  # Discount factor
        epsilon = 0.1  # Exploration rate
        
        # Training loop
        for episode in range(num_episodes):
            # Initialize state
            current_state = state
            
            # Choose action
            if np.random.rand() < epsilon:
                action = np.random.choice(range(5))  # Random action
            else:
                action = np.argmax(Q[current_state])
                
            # Execute action
            next_state, reward, done = take_action(action)
            
            # Update Q-table
            Q[current_state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[current_state][action])
            
            # Update state
            current_state = next_state
            
            # If episode is done, reset state
            if done:
                current_state = state
            
        return Q

    def rollout(self, track, Q_table, planner=None, max_frames=1000, verbose=False, data_callback=None):
        """
        Play a level (track) for a single round.
        :param track: Name of the track
        :param controller: low-level controller, see controller.py
        :param planner: high-level planner, see planner.py
        :param max_frames: Maximum number of frames to play for
        :param verbose: Should we use matplotlib to show the agent drive?
        :param data_callback: Rollout calls data_callback(time_step, image, 2d_aim_point) every step, used to store the
                              data
        :return: Number of steps played
        """
        
        
        # Hyperparameters
        alpha = 1  # Learning rate
        gamma = 0.99  # Discount factor
        epsilon = 0.1  # Exploration rate
        q_state = None
        
        
        if self.k is not None and self.k.config.track == track:
            self.k.restart()
            self.k.step()
        else:
            if self.k is not None:
                self.k.stop()
                del self.k
            config = pystk.RaceConfig(num_kart=1, laps=1,  track=track)
            config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL

            self.k = pystk.Race(config)
            self.k.start()
            self.k.step()

        state = pystk.WorldState()
        track = pystk.Track()

        last_rescue = 0
        
        if verbose:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1)
        
        for t in range(max_frames):

            state.update()
            track.update()

            kart = state.players[0].kart

            if np.isclose(kart.overall_distance / track.length, 1.0, atol=2e-3):
                if verbose:
                    print("Finished at t=%d" % t)
                    
                
                # Update Q-table
                q_next_state = [
                    np.clip(int((aim_point_image[0] + 1) * 64), 0, 127), 
                    np.clip(int((aim_point_image[1] + 1) * 48), 0, 95), 
                    np.clip(int(current_vel), 0, 30)
                ]
                reward = 1000000
                Q_table[q_state[0], q_state[1], q_state[2], :, :, :, :, :] += alpha * (reward + gamma * np.max(Q_table[q_next_state[0], q_next_state[1], q_next_state[2], :, :, :, :, :]) - Q_table[q_state[0], q_state[1], q_state[2], :, :, :, :, :])
                break

            proj = np.array(state.players[0].camera.projection).T
            view = np.array(state.players[0].camera.view).T

            aim_point_world = self._point_on_track(kart.distance_down_track+TRACK_OFFSET, track)
            aim_point_image = self._to_image(aim_point_world, proj, view)
            if data_callback is not None:
                data_callback(t, np.array(self.k.render_data[0].image), aim_point_image)

            if planner:
                image = np.array(self.k.render_data[0].image)
                aim_point_image = planner(TF.to_tensor(image)[None]).squeeze(0).cpu().detach().numpy()
            
            current_vel = np.linalg.norm(kart.velocity)    
            
            if q_state is None:
                q_state = [
                    np.clip(int((aim_point_image[0] + 1) * 64), 0, 127), 
                    np.clip(int((aim_point_image[1] + 1) * 48), 0, 95), 
                    np.clip(int(current_vel), 0, 30)
                ]
            else:
                q_next_state = [
                    np.clip(int((aim_point_image[0] + 1) * 64), 0, 127), 
                    np.clip(int((aim_point_image[1] + 1) * 48), 0, 95), 
                    np.clip(int(current_vel), 0, 30)
                ]
                reward = (kart.overall_distance / track.length - 0.5) * t
                Q_table[q_state[0], q_state[1], q_state[2], :, :, :, :, :] += alpha * (reward + gamma * np.max(Q_table[q_next_state[0], q_next_state[1], q_next_state[2], :, :, :, :, :]) - Q_table[q_state[0], q_state[1], q_state[2], :, :, :, :, :])
                q_state = q_next_state
            
            if np.random.rand() < epsilon:
                steer = np.random.choice(9)
                accel = np.random.choice(10)
                brake = np.random.choice(2)
                drift = np.random.choice(2)
                nitro = np.random.choice(2)
                action_indices = [steer, accel, brake, drift, nitro]
            else:
                action_indices = np.unravel_index(np.argmax(Q_table[q_state[0], q_state[1], q_state[2], :, :, :, :, :], axis=None), Q_table[q_state[0], q_state[1], q_state[2], :, :, :, :, :].shape)
                # action_indices = np.argmax(Q_table[q_state[0], q_state[1], q_state[2], :, :, :, :, :])
            
            action = pystk.Action()
            action.steer = action_indices[0] // 2 if action_indices[0] % 2 == 0 else -action_indices[0] // 2
            action.acceleration = action_indices[1] * 0.1 + 0.1
            action.brake = action_indices[2] == 1
            action.drift = action_indices[3] == 1
            action.nitro = action_indices[4] == 1
            
            # action = controller(aim_point_image, current_vel)

            if current_vel < 1.0 and t - last_rescue > RESCUE_TIMEOUT:
                last_rescue = t
                action.rescue = True

            if verbose:
                ax.clear()
                ax.imshow(self.k.render_data[0].image)
                WH2 = np.array([self.config.screen_width, self.config.screen_height]) / 2
                ax.add_artist(plt.Circle(WH2*(1+self._to_image(kart.location, proj, view)), 2, ec='b', fill=False, lw=1.5))
                ax.add_artist(plt.Circle(WH2*(1+self._to_image(aim_point_world, proj, view)), 2, ec='r', fill=False, lw=1.5))
                if planner:
                    ap = self._point_on_track(kart.distance_down_track + TRACK_OFFSET, track)
                    ax.add_artist(plt.Circle(WH2*(1+aim_point_image), 2, ec='g', fill=False, lw=1.5))
                plt.pause(1e-3)

            self.k.step(action)
            t += 1
        return t, kart.overall_distance / track.length, Q_table

    def close(self):
        """
        Call this function, once you're done with PyTux
        """
        if self.k is not None:
            self.k.stop()
            del self.k
        pystk.clean()
        
if __name__ == '__main__':
    from argparse import ArgumentParser
    
    tracks_limits = {
        "zengarden": 500,
        "lighthouse": 500,
        "hacienda": 700,
        "snowtuxpeak": 700,
        "cornfield_crossing": 800,
        "scotland": 800,
    }
    
    tracks_lengths = {
        "zengarden": 710.3971557617188,
        "lighthouse": 879.8558959960938,
        "hacienda": 1166.5313720703125,
        "snowtuxpeak": 992.5704956054688,
        "cornfield_crossing": 1263.839111328125,
        "scotland": 1328.66259765625,
    }
    
    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-e', '--episodes', type=int, default=1000)
    parser.add_argument('-ld', '--load', action='store_true')
    args = parser.parse_args()
    
    pytux = PyTux()
    for t in args.track:
        if args.load:
            Q_table = np.load(f"trained_models/Q_0_table_{t}.npy")
            print(f"trained_models/Q_0_table_{t}.npy loaded")
        else:
            # Initialize Q-table
            Q_table = np.zeros((128, 96, 30, 9, 10, 2, 2, 2))  # 5 actions: steer, acceleration, brake, drift, nitro
        
        for episode in range(args.episodes):
            steps, how_far, Q_table = pytux.rollout(t, Q_table, max_frames=1000, verbose=args.verbose)
            print(steps, how_far, t)
            if steps > tracks_limits[t]:
                print("❌ Missed the track limit by %d steps" % (steps - tracks_limits[t]))
            else:
                print("✅ Passed the track limit by %d steps!" % (tracks_limits[t] - steps))
            print("-" * 20)
        print(Q_table)
        np.save(f"trained_models/Q_0_table_{t}.npy", Q_table)
        print(f"trained_models/Q_0_table_{t}.npy saved")
    pytux.close()