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

CONFIG_DIR = "bin/teams/base/config/formations-dt"
SERVER_ADDR = "localhost"
TEAM_NAME = "base_left"
PLAY_GOALIE = False
RECORD_DIR = ""
# Lowered to 100000 from 10000000 as training only for dashing should not take too long
MAX_ITER = 100000
TESTING = True
Q_DISPLAY_FREQ = 1000
EVAL_GAMES = 100
UPDATE_RATIO = 0.1
DASH_PARAMS_ONLY = True

if DASH_PARAMS_ONLY:
    from ddpg_dash import *
    EVAL_FREQ = 1000
    EXPLORE_ITER = 100

else:
    from ddpg import *
    EVAL_FREQ = 10000
    EXPLORE_ITER = 1000

EPSILON = 0.1
EPSILON_EVAL = 0
OFFENSE_AGENTS = 1
OFFENSE_NPCS = 0
DEFENSE_AGENTS = 0
DEFENSE_NPCS = 0
OFFENSE_DUMMIES = 0
DEFENSE_DUMMIES = 0
DEFENSE_CHASERS = 0 

# averageQFile = open("q_values_longer_correct_sign_long_run.txt", "w+", 0)
settingsFile = open("settings_dash_only_params_iter100000_explore100_modreward2.txt", "w+", 0)
settingsFile.write(str(DASH_PARAMS_ONLY) + " " + str(MAX_ITER) + " " + str(EVAL_FREQ) + " " + str(EXPLORE_ITER))
settingsFile.close()
evalFile = open("eval_trials_longer_correct_sign_dash_only_params_iter100000_explore100_modreward2.txt", "w+", 0)
replay_buffer_file = gzip.open("replay_buffer_iter100000_explore100_modreward2.pkl.gz", "wb", 0)
# actionFile = open("actions_eval_longer_correct_sign_dash_only_params_iter100000_explore100_modreward2.txt", "w+", 0)
# trainingActionReward = open("actions_training_dash_only_2.txt", "w+", 0)
# trainingEstimatedQValues = open("training_estimated_q_values_2.txt", "w+", 0)
# distanceToBallFile = open("distance_to_ball.txt", "w+")

def calculate_epsilon(iterations):
	if iterations < EXPLORE_ITER:
		return 1.0 - (1.0 - EPSILON) * (iterations / (EXPLORE_ITER + 0.0))
	return EPSILON

def playEpisode(HFO, actorNet, criticNet, epsilon, update, tid):
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
            '''
            if not update:
                actionFile.write(str(action) + " " + "{:.8f}".format(action_arg1) + " " + "{:.8f}".format(action_arg2))
                actionFile.write("\n")
            '''
            '''
            game_state = HFO.getState()
            ball_ang_sin_rad = game_state[51]
            ball_ang_cos_rad = game_state[52]
            ball_ang_rad = np.arccos(ball_ang_cos_rad)
            if ball_ang_sin_rad < 0:
                ball_ang_rad *= -1
            '''
            # trainingActionReward.write(str(update) + " " + str(action) + " " + "{:.8f}".format(action_arg1) + " " + "{:.8f}".format(action_arg2) + " " + "{:.8f}".format(ball_ang_rad * 180 / np.pi) + " ")
            if DASH_PARAMS_ONLY:
                HFO.act(action, action_arg1, action_arg2)
            else:
                if action == 0 or action == 3:
                    HFO.act(action, action_arg1, action_arg2)
                else:
                    HFO.act(action, action_arg1)
            game.update(HFO)
            kickable = HFO.getState()[12]
            # distanceToBallFile.write(str(game.old_ball_prox))
            # distanceToBallFile.write("\n")
            reward = game.reward()
            '''
            trainingActionReward.write("{:.8f}".format(reward))
            trainingActionReward.write("\n")
            '''
            # print reward
            if update:
                next_state = HFO.getState()
                next_state_size = HFO.getStateSize()
                assert next_state_size == actorNet.state_size and next_state_size == criticNet.state_size
                transition = None
                if game.status == IN_GAME:
                    transition = (past_states[0], actor_output, reward, 0, next_state)
                else:
                    transition = (past_states[0], actor_output, reward, 0, np.zeros(actorNet.state_size))
                episode.append(transition)
    if update:
        actorNet.replay_buffer.addWithLabel(episode)
        '''
        for i in xrange(len(episode)):
            trainingEstimatedQValues.write("{:.8f}".format(episode[i][2]) + " " + "{:.8f}".format(episode[i][3]))
            trainingEstimatedQValues.write("\n")
        '''
    # print "Total Reward: ", game.total_reward
    # assert game.total_reward != 0
    '''
    else:
        actionFile.write(str(actorNet.iterations) + "\n\n")
    '''
    return game.total_reward, game.steps, game.status, game.extrinsic_reward, initial_dist, kickable

def keepPlaying(tid, port):
    num_players = OFFENSE_AGENTS + OFFENSE_NPCS + DEFENSE_AGENTS + DEFENSE_NPCS + OFFENSE_DUMMIES + DEFENSE_DUMMIES + DEFENSE_CHASERS
    num_features = num_state_features(num_players)
    with tf.Session() as sess:
        if DASH_PARAMS_ONLY:
            actor = ActorNetwork(sess, num_features, DASH_PARAM_SIZE, tid, \
                ACTOR_LEARNING_RATE, TAU)
            critic = CriticNetwork(sess, num_features, DASH_PARAM_SIZE, tid, \
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
        saver = tf.train.Saver(max_to_keep=None)
        actor.unum = Player._fields_[1][1]
        critic.unum = Player._fields_[1][1]
        num_iter = max(actor.iterations, critic.iterations)
        episode = 0
        eval_iter = max(actor.iterations, critic.iterations)
        num_eval = 0
        while max(actor.iterations, critic.iterations) < MAX_ITER:
            epsilon = calculate_epsilon(max(actor.iterations, critic.iterations))
            total_reward, steps, status, extrinsic_reward, initial_dist, kickable = playEpisode(HFO, actor, critic, epsilon, True, tid)
            n_updates = int(np.floor(steps * UPDATE_RATIO))
            ave_q = 0
            for i in xrange(n_updates):
                update_q = update(sess, actor.replay_buffer, actor, critic)
                if TESTING and update_q != None:
                    ave_q += (update_q / Q_DISPLAY_FREQ)
                    # print actor.iterations
                    if actor.iterations % Q_DISPLAY_FREQ == 0:
                        # print 5
                        # averageQFile.write(str(actor.iterations) + " " + "{:.8f}".format(ave_q))
                        # averageQFile.write("\n")
                        ave_q = 0
            episode += 1
            if TESTING and actor.iterations >= eval_iter + EVAL_FREQ:
                num_eval += 1
                if num_eval % 10 == 0 or actor.iterations + EVAL_FREQ >= MAX_ITER:
                    save_path = saver.save(sess, "Models100000reward2/model" + str(actor.iterations) + ".ckpt")
                    if actor.iterations + EVAL_FREQ >= MAX_ITER:
                        cPickle.dump(actor.replay_buffer, replay_buffer_file, -1)
                eval_game_rewards = []
                eval_game_steps = []
                goals = 0
                eval_game_success_steps = []
                eval_kickable = []
                # eval_game_dist_proportion_travelled = []
                for j in xrange(EVAL_GAMES):
                    eval_reward, eval_steps, eval_status, eval_ext_reward, initial_dist, kickable = playEpisode(HFO, actor, critic, EPSILON_EVAL, False, tid)
                    # print "{:.16f}".format(eval_reward), eval_steps
                    eval_game_rewards.append(eval_reward)
                    eval_game_steps.append(eval_steps)
                    eval_kickable.append(kickable)
                    # eval_game_dist_proportion_travelled.append(eval_reward / initial_dist)
                    if eval_status == GOAL:
                        goals += 1
                        eval_game_success_steps.append(eval_steps)
                avg_reward = np.mean(eval_game_rewards)
                reward_stddev = np.std(eval_game_rewards)
                avg_kickable = np.mean(eval_kickable)
                '''
                avg_dist_proportion = np.mean(eval_game_dist_proportion_travelled)
                dist_proportion_stddev = np.std(eval_game_dist_proportion_travelled)
                '''
                avg_steps = np.mean(eval_game_steps)
                steps_stddev = np.std(eval_game_steps)
                avg_success_steps = 0
                success_steps_std = 0
                if len(eval_game_success_steps) > 0:
                    avg_success_steps = np.mean(eval_game_success_steps)
                    success_steps_std = np.std(eval_game_success_steps)
                percent_goal = (goals + 0.0) / EVAL_GAMES
                evalFile.write(str(actor.iterations) + " " + "{:.8f}".format(avg_reward) + " " + "{:.8f}".format(reward_stddev) + " " + \
                    # "{:.8f}".format(avg_dist_proportion) + " " + "{:.8f}".format(dist_proportion_stddev) + " " + \
                    "{:.4f}".format(avg_kickable) + " " + \
                    "{:.4f}".format(avg_steps) + " " + "{:.4f}".format(steps_stddev) + " " + "{:.4f}".format(avg_success_steps) + " " + \
                    "{:.4f}".format(success_steps_std) + " " + "{:.4f}".format(percent_goal))
                evalFile.write("\n")
                eval_iter = actor.iterations
        
        replay_buffer_file.close()
        evalFile.close()
        # averageQFile.close()
        # actionFile.close()
        # trainingActionReward.close()
        # trainingEstimatedQValues.close()
        # distanceToBallFile.close()
        HFO.act(QUIT)
        HFO.step()

def main():
    keepPlaying(0, 6000)
    '''
    num_players = OFFENSE_AGENTS + OFFENSE_NPCS + DEFENSE_AGENTS + DEFENSE_NPCS + OFFENSE_DUMMIES + DEFENSE_DUMMIES + DEFENSE_CHASERS
    num_features = num_state_features(num_players)
    HFO = HFOEnvironment()
    HFO.connectToServer(LOW_LEVEL_FEATURE_SET, CONFIG_DIR, 6000, SERVER_ADDR, TEAM_NAME, PLAY_GOALIE, RECORD_DIR)
    time.sleep(5)
    game = HFOState(11)
    HFO.act(DASH, 0, 0)
    game.update(HFO)
    HFO.act(0, 100, 0)
    game.update(HFO)
    print game.reward()
    HFO.act(0, 100, 90)
    game.update(HFO)
    print game.reward()
    HFO.act(0, 100, 180)
    game.update(HFO)
    print game.reward()
    HFO.act(0, 100, 270)
    game.update(HFO)
    print game.reward()
    HFO.act(0, 50, 0)
    game.update(HFO)
    print game.reward()
    game_state = HFO.getState()
    ball_ang_sin_rad = game_state[51]
    ball_ang_cos_rad = game_state[52]
    ball_ang_rad = np.arccos(ball_ang_cos_rad)
    if ball_ang_sin_rad < 0:
        ball_ang_rad *= -1
    HFO.act(0, 100, 180 / np.pi * ball_ang_rad)
    game.update(HFO)
    print game.reward()
    '''
if __name__ == '__main__':
	main()
