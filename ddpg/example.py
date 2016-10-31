from gym_soccer_interface import *
import numpy as np

game = SoccerEnv()
game._render()


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

    while dist > 1e-3:
        feature = game.env.getState()
        sin = feature[51]
        cos = feature[52]
        dist = feature[53]
        ball = feature[9]
        angle = 180. / np.pi * np.sign(sin) * np.arccos(cos)
        turn_action = [1, 0, 0, angle, 0, 0]
        run_action = [0, 50, 0, 0, 0, 0]
        game._step(turn_action)
        game._step(run_action)

    game._reset
