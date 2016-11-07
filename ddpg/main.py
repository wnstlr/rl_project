import tensorflow as tf
import tflearn
import numpy as np
import gym
import time

from gym_soccer.envs.soccer_env import *
from gym_soccer.envs.soccer_empty_goal import *
from gym_soccer.envs.soccer_against_keeper import *

from ctypes import *
import numpy as np
from numpy.ctypeslib import as_ctypes

from ddpg import *
from ReplayBuffer import *

from gym_soccer_interface import *

class Player(Structure): pass
Player._fields_ = [
    ('side', c_int),
    ('unum', c_int),
]

def calculate_epsilon(iterations):
	if iterations < EXPLORE_ITER:
		return 1.0 - (1.0 - EPSILON) * (iterations / EXPLORE_ITER)
	return EPSILON

def playEpisode(GymSoccer, actorNet, criticNet, epsilon, update, tid):
    episode = []
    game = GymSoccer
    game.env._step([0, 20, -1, 0, 0, 0])
    game.update()
    past_states = deque()
    assert not game.episode_over
    while not game.episode_over:
        current_state = game.env.env.getState()
        current_state_size = game.env.env.getStateSize()
        #print current_state.size
        #print actorNet.state_size
        assert current_state_size == actorNet.state_size and current_state_size == criticNet.state_size
        past_states.append(current_state)
        if len(past_states) < STATE_INPUT_COUNT:
            game.env._step([0, 20, -1, 0, 0, 0])
        else:
            while len(past_states) > STATE_INPUT_COUNT:
                past_states.popleft()
                actor_output = actorNet.select_action(past_states[0], epsilon)
                # print actor_output.shape
                # print actor_output
                actor_output_row = actor_output[0]
                print actor_output.shape
                print actor_output_row.shape
                action, action_arg1, action_arg2 = get_action(actor_output_row)

                if action == 0: # Dash
                    action_vec = [action, action_arg1, action_arg2, 0, 0, 0]
                elif action == 1: # Turn
                    action_vec = [action, 0, 0, action_arg1, 0, 0]
                elif action == 2: # Tackle
                    action_vec = [3, action_arg1]
                elif action == 3: # Kick
                    action_vec = [2, 0, 0, 0, action_arg1, action_arg2]

                game.env._step(action_vec)
                game.update()
                reward = game.reward()
                if update:
                    next_state = game.env.env.getState()
                    next_state_size = game.env.env.getStateSize()
                    assert next_state_size == actorNet.state_size and next_state_size == criticNet.state_size
                    transition = None
                    if game.status == hfo_py.IN_GAME:
                        transition = (past_states[0], actor_output_row, reward, 0, next_state)
                    else:
                        transition = (past_states[0], actor_output_row, reward, 0, np.zeros(actorNet.state_size))
                    episode.append(transition)
    if update:
        actorNet.replay_buffer.label(episode)
        actorNet.replay_buffer.addMultiple(episode)
    return game.total_reward, game.steps, game.status, game.extrinsic_reward

def num_state_features(num_players):
    return 58 + 8 * num_players

def keepPlaying(tid, port):
    num_players = OFFENSE_AGENTS + OFFENSE_NPCS + DEFENSE_AGENTS + DEFENSE_NPCS + OFFENSE_DUMMIES + DEFENSE_DUMMIES + DEFENSE_CHASERS
    num_features = num_state_features(num_players)
    with tf.Session() as sess:
        actor = ActorNetwork(sess, num_features, ACTION_SIZE, PARAM_SIZE, tid, \
            ACTOR_LEARNING_RATE, TAU)
        critic = CriticNetwork(sess, num_features, ACTION_SIZE, PARAM_SIZE, tid, \
            CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())

        GymSoccer = GymSoccerAgainstKeeperState(0)
        sess.run(tf.initialize_all_variables())
        actor.unum = Player._fields_[1][1]
        critic.unum = Player._fields_[1][1]
        num_iter = max(actor.iterations, critic.iterations)
        episode = 0
        while max(actor.iterations, critic.iterations) < MAX_ITER:
            if GymSoccer.episode_over:
                GymSoccer.episode_over = False
                GymSoccer.env._reset()
            epsilon = calculate_epsilon(max(actor.iterations, critic.iterations))
            (total_reward, steps, status, extrinsic_reward) = playEpisode(GymSoccer, actor, critic, epsilon, True, tid)
            n_updates = int(np.floor(steps * UPDATE_RATIO))
            for i in xrange(n_updates):
                update(sess, actor.replay_buffer, actor, critic)
            #GymSoccer.env._reset()

def main():
	keepPlaying(0, 6000)

if __name__ == '__main__':
	main()
