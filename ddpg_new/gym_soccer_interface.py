import gym
import numpy as np

from gym_soccer.envs.soccer_env import *
from gym_soccer.envs.soccer_empty_goal import *
from gym_soccer.envs.soccer_against_keeper import *

pass_vel_threshold = -0.5

class GymSoccerAgainstKeeperState():
    def __init__(self, unum):
        self.old_ball_prox = 0
        self.ball_prox_change = 0
        self.old_kickable = 0
        self.kickable_change = 0
        self.old_ball_goal_dist = 0
        self.ball_goal_dist_change = 0
        self.steps = 0
        self.total_reward = 0
        self.extrinsic_reward = 0
        self.status = hfo_py.IN_GAME
        self.episode_over = False
        self.got_kickable_reward = False
        self.unum = unum
        self.pass_active = 0
        self.pass_vel_threshold = -0.5
        self.old_player_on_ball = None
        self.player_on_ball = None

    def create_env(self):
        self.env = SoccerEnv()
        #self.env = SoccerAgainstKeeperEnv()
        self.env._render()
        self.update()

    def reset(self, unum):
        self.env._reset()
        self.old_ball_prox = 0
        self.ball_prox_change = 0
        self.old_kickable = 0
        self.kickable_change = 0
        self.old_ball_goal_dist = 0
        self.ball_goal_dist_change = 0
        self.steps = 0
        self.total_reward = 0
        self.extrinsic_reward = 0
        self.status = hfo_py.IN_GAME
        self.episode_over = False
        self.got_kickable_reward = False
        self.unum = unum
        self.pass_active = 0
        self.pass_vel_threshold = -0.5
        self.old_player_on_ball = None
        self.player_on_ball = None
        self.update()

    def update(self):
        status = self.env.status
        if status == hfo_py.SERVER_DOWN:
            print "Server Down!"
            sys.exit(1)
        elif status != hfo_py.IN_GAME:
            self.episode_over = True
        game_state = self.env.env.getState()
        ball_proximity = game_state[53]
        goal_proximity = game_state[15]
        ball_dist = 1.0 - ball_proximity
        goal_dist = 1.0 - goal_proximity
        kickable = game_state[12]
        ball_ang_sin_rad = game_state[51]
        ball_ang_cos_rad = game_state[52]
        ball_ang_rad = np.arccos(ball_ang_cos_rad)
        if ball_ang_sin_rad < 0:
            ball_ang_rad *= -1
        goal_ang_sin_rad = game_state[13]
        goal_ang_cos_rad = game_state[14]
        goal_ang_rad = np.arccos(goal_ang_cos_rad)
        if goal_ang_sin_rad < 0:
            goal_ang_rad *= -1
        goal_ball_ang = max(ball_ang_rad, goal_ang_rad) - min(ball_ang_rad, goal_ang_rad)
        ball_goal_dist = np.sqrt(ball_dist * ball_dist + goal_dist * goal_dist - 2 * ball_dist * goal_dist * np.cos(goal_ball_ang))
        ball_vel_valid = game_state[54]
        ball_vec = game_state[55]
        if ball_vel_valid and ball_vec > pass_vel_threshold:
            self.pass_active = True
        if self.steps > 0:
            self.ball_prox_change = ball_proximity - self.old_ball_prox
            self.kickable_change = kickable - self.old_kickable
            self.ball_goal_dist_change = ball_goal_dist - self.old_ball_goal_dist
        self.old_ball_prox = ball_proximity
        self.old_kickable = kickable
        self.old_ball_goal_dist = ball_goal_dist
        if self.episode_over:
            self.ball_prox_change = 0
            self.kickable_change = 0
            self.ball_goal_dist_change = 0
        self.old_player_on_ball = self.player_on_ball
        self.player_on_ball = self.env.env.playerOnBall()
        self.steps += 1

    def move_to_ball_reward(self):
        reward = 0
        if self.player_on_ball.unum < 0 or self.player_on_ball.unum == self.unum:
            reward += self.ball_prox_change
        if self.kickable_change >= 1 and not self.got_kickable_reward:
            reward += 1.0
            self.got_kickable_reward = True
        return reward

    def kick_to_goal_reward(self):
        if self.player_on_ball.unum == self.unum:
            return -1 * self.ball_goal_dist_change
        elif self.got_kickable_reward:
            return -0.2 * self.ball_goal_dist_change
        return 0

    def EOT_reward(self):
        if self.status == hfo_py.GOAL:
            assert self.old_player_on_ball.side == LEFT
            if self.player_on_ball.unum == self.unum:
                return 5
            else:
                return 1
        elif self.status == hfo_py.CAPTURED_BY_DEFENSE:
            return 0
        return 0

    def pass_reward(self):
        if self.pass_active and self.player_on_ball.unum > 0 and self.player_on_ball.unum != self.old_player_on_ball.unum:
            self.pass_active = False
            return 1
        return 0

    def reward(self):
        # Simple reward of reaching the ball
        return self.move_to_ball_reward()

        move_to_ball_reward = self.move_to_ball_reward()
        kick_to_goal_reward = 3 * self.kick_to_goal_reward()
        pass_reward = 3 * self.pass_reward()
        eot_reward = self.EOT_reward()
        reward = move_to_ball_reward + kick_to_goal_reward + eot_reward
        self.extrinsic_reward += eot_reward
        self.total_reward += reward
        return reward

    def get_state(self):
        return self.env.env.getState()

    def get_state_size(self):
        return len(self.get_state())

    def act(self, action, param1, param2):
        # Take an action using action and params given.
        if action == 0: # DASH
            action_vec = [0, param1, param2, 0, 0, 0]
        elif action == 1: # TURN
            action_vec = [1, 0, 0, param1, 0, 0]
        elif action == 2: # TACKLE
            action_vec = [3, 0, 0, 0, 0, 0]
        elif action == 3: # KICK
            action_vec = [2, 0, 0, 0, param1, param2]
        return self.env._step(action_vec)
