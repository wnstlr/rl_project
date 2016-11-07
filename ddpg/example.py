from gym_soccer_interface import *
import numpy as np
import pickle

game = SoccerEnv()
game._render()

feats = []

"grab ball using angle"
while True:
    feature = game.env.getState()
    sin = feature[51]
    cos = feature[52]
    dist = feature[53]
    ball = feature[9]
    angle = 180. / np.pi * np.sign(sin) * np.arccos(cos)

    turn_action = [1, 0, 0, angle, 0, 0]
    run_action = [0, 50, 0, 0, 0, 0]
    game._step(turn_action)

    #while dist < 1:
    while dist < 1:
        feature = game.env.getState()
        sin = feature[51]
        cos = feature[52]
        dist = feature[53]
        print dist
        ball = feature[9]
        angle = 180. / np.pi * np.sign(sin) * np.arccos(cos)
        turn_action = [1, 0, 0, angle, 0, 0]
        run_action = [0, 50, 0, 0, 0, 0]
        game._step(turn_action)
        game._step(run_action)

        ## Save the feature and results
        reward = 0
        vec = np.concatenate((feature, np.array([reward])))
        feats.append(vec)

    print "got ball"

    # goal_sin = feature[13]
    # goal_cos = feature[14]
    # goal_dist = feature[15]
    # goal_angle = 180. / np.pi * np.sign(goal_sin) * np.arccos(goal_cos)
    #
    # turn_action = [1, 0, 0, goal_angle, 0, 0]
    # game._step(turn_action)
    #
    # while goal_dist < 0.6:
    #     feature = game.env.getState()
    #     goal_dist = feature[15]
    #     run_action = [0, 50, 0, 0, 0, 0]
    #     game._step(run_action)
    #
    # kick_action = [2, 0, 0, 0, 50, 0]
    # game._step(kick_action)

    reward = 1
    vec = np.concatenate((feature, np.array([reward])))
    feats.append(vec)
    game._reset()

    if len(feats) > 100000:
        feats = np.array(feats)
        pickle.dump(feats, open('feats.pkl', 'wb'))
        break
