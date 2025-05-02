import pystk

"""
This is a simple controller for the Tux Racer game.
Need to specify the action to take in the game by setting these values:
    1. action.steer:         normalized angle [-1, 1]
    2. action.acceleration:  normalized [0, 1]
    3. action.brake:         bool
    4. action.drift:         bool
    5. action.nitro:         bool
"""
def control(aim_point, current_vel, steer_gain=6, skid_thresh=0.2, target_vel=30):
    import numpy as np
    #this seems to initialize an object
    action = pystk.Action()

   
 
    #compute steering
    action.steer = np.clip(10 * aim_point[0], -1, 1)
    # action.steer = 1 if aim_point[0] > 0 else -1 if aim_point[0] < 0 else 0

    #compute acceleration
    if abs(aim_point[0]) < 1 - skid_thresh:
        action.acceleration = np.clip((target_vel - current_vel)/10, 0, 1)
    else:
        action.acceleration = 0.1  # slow down in steep turns
    
    #compute brake
    if abs(aim_point[0]) > 1 - skid_thresh:
        action.brake = True
    else:
        action.brake = False
        
    #compute drift
    if abs(aim_point[0]) > 1 - skid_thresh:
        action.drift = True
    else:
        action.drift = False
    
    #compute nitro
    if abs(aim_point[0]) <= skid_thresh:
        action.nitro = True
    else:
        action.nitro = False

    return action

    




if __name__ == '__main__':
    from utils import PyTux
    from argparse import ArgumentParser
    
    print("PyTux controller")
    tracks_limits = {
        "zengarden": 500,
        "lighthouse": 500,
        "hacienda": 700,
        "snowtuxpeak": 700,
        "cornfield_crossing": 800,
        "scotland": 800,
    }

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far, t)
            if steps > tracks_limits[t]:
                print("❌ Missed the track limit by %d steps" % (steps - tracks_limits[t]))
            else:
                print("✅ Passed the track limit by %d steps!" % (tracks_limits[t] - steps))
            print("-" * 20)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
