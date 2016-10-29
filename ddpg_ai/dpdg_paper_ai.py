from hfo import *
import tensorflow as tf
import numpy as np
import tflearn
import time
from collections import deque

from replay_buffer import ReplayBuffer
from ddpg import *
from hfo_interface import *

CONFIG_DIR = "bin/teams/base/config/formations-dt"
SERVER_ADDR = "localhost"
TEAM_NAME = "base_left"
PLAY_GOALIE = False
RECORD_DIR = ""
MAX_ITER = 10000000
UPDATE_RATIO = 0.1

EPSILON = 0.1
OFFENSE_AGENTS = 1
OFFENSE_NPCS = 0
DEFENSE_AGENTS = 0
DEFENSE_NPCS = 0
OFFENSE_DUMMIES = 0
DEFENSE_DUMMIES = 0
DEFENSE_CHASERS = 0 

def calculate_epsilon(iterations):
	if iterations < EXPLORE_ITER:
		return 1.0 - (1.0 - EPSILON) * (iterations / EXPLORE_ITER)
	return EPSILON

def playEpisode(HFO, actorNet, criticNet, epsilon, update, tid):
    episode = []
    game = HFOState(actorNet.unum)
    HFO.act(DASH, 0, 0)
    game.update(HFO)
    past_states = deque()
    assert not game.episode_over
    while not game.episode_over:
        current_state = HFO.getState()
        current_state_size = HFO.getStateSize()
        # print current_state.size
        # print actorNet.state_size
        assert current_state_size == actorNet.state_size and current_state_size == criticNet.state_size
        past_states.append(current_state)
        if len(past_states) < STATE_INPUT_COUNT:
            HFO.act(DASH, 0, 0)
        else:
            while len(past_states) > STATE_INPUT_COUNT:
                past_states.popleft()
            actor_output = actorNet.select_action(past_states[0], epsilon)
            # print actor_output.shape
            # print actor_output
            actor_output_row = actor_output[0]
            action, action_arg1, action_arg2 = get_action(actor_output_row)
            if action == 0 or action == 3:
                HFO.act(action, action_arg1, action_arg2)
            else:
                HFO.act(action, action_arg1)
            game.update(HFO)
            reward = game.reward()
            if update:
                next_state = HFO.getState()
                next_state_size = HFO.getStateSize()
                assert next_state_size == actorNet.state_size and next_state_size == criticNet.state_size
                transition = None
                if game.status == IN_GAME:
                    transition = (past_states[0], actor_output_row, reward, 0, next_state)
                else:
                    transition = (past_states[0], actor_output_row, reward, 0, np.zeros(actorNet.state_size))
                episode.append(transition)
    if update:
        actorNet.replay_buffer.label(episode)
        actorNet.replay_buffer.addMultiple(episode)
    return game.total_reward, game.steps, game.status, game.extrinsic_reward

def keepPlaying(tid, port):
    num_players = OFFENSE_AGENTS + OFFENSE_NPCS + DEFENSE_AGENTS + DEFENSE_NPCS + OFFENSE_DUMMIES + DEFENSE_DUMMIES + DEFENSE_CHASERS
    num_features = num_state_features(num_players - 1)
    with tf.Session() as sess:
        actor = ActorNetwork(sess, num_features, ACTION_SIZE, PARAM_SIZE, tid, \
            ACTOR_LEARNING_RATE, TAU)
        critic = CriticNetwork(sess, num_features, ACTION_SIZE, PARAM_SIZE, tid, \
            CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())
        HFO = HFOEnvironment()
        HFO.connectToServer(LOW_LEVEL_FEATURE_SET, CONFIG_DIR, port, SERVER_ADDR, TEAM_NAME, PLAY_GOALIE, RECORD_DIR)
        time.sleep(5)
        sess.run(tf.initialize_all_variables())
        actor.unum = Player._fields_[1][1]
        critic.unum = Player._fields_[1][1]
        num_iter = max(actor.iterations, critic.iterations)
        episode = 0
        while max(actor.iterations, critic.iterations) < MAX_ITER:
            epsilon = calculate_epsilon(max(actor.iterations, critic.iterations))
            (total_reward, steps, status, extrinsic_reward) = playEpisode(HFO, actor, critic, epsilon, True, tid)
            n_updates = int(np.floor(steps * UPDATE_RATIO))
            for i in xrange(n_updates):
                update(sess, actor.replay_buffer, actor, critic)
        HFO.act(QUIT)
        HFO.step()

def main():
	keepPlaying(0, 6000)

if __name__ == '__main__':
	main()
		
