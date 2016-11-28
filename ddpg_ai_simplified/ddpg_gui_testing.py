from hfo import *
import tensorflow as tf
import numpy as np
import tflearn
import time
from collections import deque

from replay_buffer import ReplayBuffer
from hfo_interface import *
import cPickle
import gzip
import pprint

CONFIG_DIR = "bin/teams/base/config/formations-dt"
SERVER_ADDR = "localhost"
TEAM_NAME = "base_left"
PLAY_GOALIE = False
RECORD_DIR = ""
EVAL_GAMES = 100
DASH_PARAMS_ONLY = False
DASH_TURN_ONLY = True
LOAD_PATH = "Models_and_data/Models60000_dashturn_try2_1000explore_no_x_limits_30000restore/model57977.ckpt"

if DASH_PARAMS_ONLY:
    from ddpg_dash import *

elif DASH_TURN_ONLY:
    from ddpg_dash_turn import *

else:
    from ddpg import *

EPSILON = 0.1
EPSILON_EVAL = 0
OFFENSE_AGENTS = 1
OFFENSE_NPCS = 0
DEFENSE_AGENTS = 0
DEFENSE_NPCS = 0
OFFENSE_DUMMIES = 0
DEFENSE_DUMMIES = 0
DEFENSE_CHASERS = 0 


def playEpisode(HFO, actorNet, criticNet, epsilon, tid):
    episode = []
    game = HFOState(actorNet.unum)
    HFO.act(DASH, 0, 0)
    game.update(HFO)
    past_states = deque()
    assert not game.episode_over
    initial_step = True
    initial_dist = 0
    kickable = 0
    while not game.episode_over:
        current_state = HFO.getState()
        current_state_size = HFO.getStateSize()
        if initial_step:
            initial_step = False
            initial_dist = current_state[53]
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
            if DASH_PARAMS_ONLY:
                action, action_arg1, action_arg2 = 0, actor_output[0], actor_output[1]
            else:
                action, action_arg1, action_arg2 = get_action(actor_output)
            if DASH_PARAMS_ONLY:
                HFO.act(action, action_arg1, action_arg2)
            else:
                if action == 0 or action == 3:
                    HFO.act(action, action_arg1, action_arg2)
                else:
                    HFO.act(action, action_arg1)
            game.update(HFO)
            kickable = HFO.getState()[12]
    return game.total_reward, game.steps, game.status, game.extrinsic_reward, initial_dist, kickable

def keepPlaying(tid, port):
    num_players = OFFENSE_AGENTS + OFFENSE_NPCS + DEFENSE_AGENTS + DEFENSE_NPCS + OFFENSE_DUMMIES + DEFENSE_DUMMIES + DEFENSE_CHASERS
    num_features = num_state_features(num_players)
    '''
    pkl_file = gzip.open("replay_buffer_iter100000_explore100_modreward2.pkl.gz", "rb")
    data1 = cPickle.load(pkl_file)
    pprint.pprint(data1)
    pkl_file.close()
    '''
    with tf.Session() as sess:
        if DASH_PARAMS_ONLY:
            actor = ActorNetwork(sess, num_features, DASH_PARAM_SIZE, tid, \
                ACTOR_LEARNING_RATE, TAU)
            critic = CriticNetwork(sess, num_features, DASH_PARAM_SIZE, tid, \
                CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())
        elif DASH_TURN_ONLY:
            actor = ActorNetwork(sess, num_features, ACTION_SIZE, tid, \
                ACTOR_LEARNING_RATE, TAU)
            critic = CriticNetwork(sess, num_features, ACTION_SIZE, tid, \
                CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())
        else:
            actor = ActorNetwork(sess, num_features, ACTION_SIZE, PARAM_SIZE, tid, \
                ACTOR_LEARNING_RATE, TAU)
            critic = CriticNetwork(sess, num_features, ACTION_SIZE, PARAM_SIZE, tid, \
                CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())
        HFO = HFOEnvironment()
        HFO.connectToServer(LOW_LEVEL_FEATURE_SET, CONFIG_DIR, port, SERVER_ADDR, TEAM_NAME, PLAY_GOALIE, RECORD_DIR)
        time.sleep(5)
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        saver.restore(sess, LOAD_PATH)
        actor.unum = Player._fields_[1][1]
        critic.unum = Player._fields_[1][1]
        num_iter = max(actor.iterations, critic.iterations)
        episode = 0
        eval_iter = max(actor.iterations, critic.iterations)
        for i in xrange(EVAL_GAMES):
            eval_reward, eval_steps, eval_status, eval_ext_reward, initial_dist, kickable = playEpisode(HFO, actor, critic, EPSILON_EVAL, tid)
            print kickable
        HFO.act(QUIT)
        HFO.step()

def main():
    keepPlaying(0, 6000)

if __name__ == '__main__':
	main()
