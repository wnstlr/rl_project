""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow

Author: Patrick Emami
"""
import tensorflow as tf
import numpy as np
import gym 
import tflearn

from replay_buffer import ReplayBuffer

# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 50000
# Max episode length
MAX_EP_STEPS = 1000
MOMENTUM = 0.95
MOMENTUM_2 = 0.999
MAX_NORM = 10
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.00001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor 
GAMMA = 0.99
# Soft target update param
TAU = 0.001
EXPLORE_ITER = 10000
BETA = 0.5
STATE_INPUT_COUNT = 1



# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = True
# Use Gym Monitor
GYM_MONITOR_EN = True
# Gym environment
ENV_NAME = 'Pendulum-v0'
# Directory for storing gym results
MONITOR_DIR = './results/gym_ddpg'
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
# RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 500000
MEMORY_THRESHOLD = 1000
MINIBATCH_SIZE = 32
ACTION_SIZE = 4
PARAM_SIZE = 6
ACTION_INPUT_DATA_SIZE = MINIBATCH_SIZE * ACTION_SIZE
ACTION_PARAMS_INPUT_DATA_SIZE = MINIBATCH_SIZE * PARAM_SIZE
TARGET_INPUT_DATA_SIZE = MINIBATCH_SIZE * ACTION_SIZE
FILTER_INPUT_DATA_SIZE = MINIBATCH_SIZE * ACTION_SIZE


# ===========================
#   Actor and Critic DNNs
# ===========================
class ActorNetwork(object):
    """ 
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -2 and 2
    """
    def __init__(self, sess, state_size, action_dim, action_param_dim, tid, learning_rate, tau):
        self.sess = sess
        self.state_size = state_size
        self.state_input_data_size = MINIBATCH_SIZE * state_size
        self.action_dim = action_dim
        self.action_param_dim = action_param_dim
        self.tid = tid
        self.learning_rate = learning_rate
        self.tau = tau
        self.unum = 0
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, GAMMA)
        self.iterations = 0

        # Actor Network
        self.inputs, self.action_out, self.action_param_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_action_out, self.target_action_param_out = self.create_actor_network()
        
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.mul(self.network_params[i], self.tau) + \
                tf.mul(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.action_dim])
        self.action_param_gradient = tf.placeholder(tf.float32, [None, self.action_param_dim])
        
        # Combine the gradients here 
        assert self.action_out != None
        assert self.network_params != None
        assert self.action_gradient != None
        self.actor_gradients = tf.gradients(self.action_out, self.network_params, -self.action_gradient)
        self.actor_param_gradients = tf.gradients(self.action_param_out, self.network_params, -self.action_param_gradient)

        # Optimization Op
        # Not sure how to clip gradients to norm 10, I get a ValueException: None Values Not Supported error
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1 = MOMENTUM, beta2 = MOMENTUM_2)
        assert self.actor_gradients != None
        assert MAX_NORM != None
        self.optimize_action = self.optimizer.apply_gradients(zip(self.actor_gradients, self.network_params))
        self.optimize_action_param = self.optimizer.apply_gradients(zip(self.actor_param_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self): 
        inputs = tflearn.input_data(shape=[None, self.state_size])
        w_init = tflearn.initializations.normal(stddev = 0.01)
        net = tflearn.fully_connected(inputs, 1024, weights_init=w_init)
        net = tflearn.activation(tflearn.activations.leaky_relu(net, 0.01))
        
        w_init = tflearn.initializations.normal(stddev = 0.01)
        net = tflearn.fully_connected(net, 512, weights_init = w_init)
        net = tflearn.activation(tflearn.activations.leaky_relu(net, 0.01))
        
        w_init = tflearn.initializations.normal(stddev = 0.01)
        net = tflearn.fully_connected(net, 256, weights_init = w_init)
        net = tflearn.activation(tflearn.activations.leaky_relu(net, 0.01))
        
        w_init = tflearn.initializations.normal(stddev = 0.01)
        net = tflearn.fully_connected(net, 128, weights_init = w_init)
        net = tflearn.activation(tflearn.activations.leaky_relu(net, 0.01))
 
        # Final layer weights are init to Normal(0, 0.01)
        w_init = tflearn.initializations.normal(stddev = 0.01)
        action_out = tflearn.fully_connected(net, self.action_dim, weights_init = w_init)
        
        w_init = tflearn.initializations.normal(stddev = 0.01)
        action_param_out = tflearn.fully_connected(net, self.action_param_dim, weights_init = w_init)
        return inputs, action_out, action_param_out

    def train(self, inputs, a_gradient, a_param_gradient):
        self.sess.run([self.optimize_action, self.optimize_action_param], feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient,
            self.action_param_gradient: a_param_gradient
        })

    def predict(self, inputs):
        return self.sess.run([self.action_out, self.action_param_out], feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run([self.target_action_out, self.target_action_param_out], feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
    
    def get_random_actor_output(self):
        random_action = np.zeros(ACTION_SIZE + PARAM_SIZE)
        for i in xrange(ACTION_SIZE):
            random_action[i] = np.random.uniform(-1.0, 1.0)
        random_action[ACTION_SIZE] = np.random.uniform(-100.0, 100.0)
        random_action[ACTION_SIZE + 1] = np.random.uniform(-180.0, 180.0)
        random_action[ACTION_SIZE + 2] = np.random.uniform(-180.0, 180.0)
        random_action[ACTION_SIZE + 3] = np.random.uniform(-180.0, 180.0)
        random_action[ACTION_SIZE + 4] = np.random.uniform(0.0, 100.0)
        random_action[ACTION_SIZE + 5] = np.random.uniform(-180.0, 180.0)
        return random_action

    def select_action(self, state, epsilon):
        return self.select_actions([state], epsilon)

    def select_actions(self, states, epsilon):
        assert epsilon >= 0.0 and epsilon <= 1.0
        assert len(states) <= MINIBATCH_SIZE
        if np.random.uniform() < epsilon:
            actor_output = np.zeros((len(states), ACTION_SIZE + PARAM_SIZE))
            for i in xrange(len(actor_output)):
                actor_output[i] = self.get_random_actor_output()
            return actor_output
        action_list = self.predict(states)
        actor_output= np.zeros((len(states), ACTION_SIZE + PARAM_SIZE))
        actor_output = np.concatenate((action_list[0], action_list[1]), 1)
        return actor_output
			
    def sample_action(self, actor_output):
        dash_prob = max(0, actor_output[0])
        turn_prob = max(0, actor_output[1])
        """
        Set tackle probability to 0 since there is no use to tackling in 1 vs 0 situation.
        """
        tackle_prob = 0
        kick_prob = max(0, actor_output[3])
        action_type = np.random.choice(3, p = [dash_prob, turn_prob, tackle_prob, 0])
        arg1 = self.get_param(self, action_type, 0)
        assert arg1 >= 0
        action_arg1 = actor_output[ACTION_SIZE + arg1]
        arg2 = self.get_param(self, action_type, 1)
        action_arg2 = 0
        if arg2 < 0:
            action_arg2 = 0
        else:
            action_arg2 = actor_output[ACTION_SIZE + arg2]
        return action_type, action_arg1, action_arg2
		

class CriticNetwork(object):
    """ 
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """
    def __init__(self, sess, state_size, action_dim, action_param_dim, tid, learning_rate, tau, num_actor_vars):
        self.sess = sess
        self.state_size = state_size
        self.action_dim = action_dim
        self.action_param_dim = action_param_dim
        self.tid = tid
        self.learning_rate = learning_rate
        self.tau = tau
        self.unum = 0
        self.iterations = 0

        # Create the critic network
        self.inputs, self.action, self.action_params, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_action_params, self.target_out = self.create_critic_network()
        
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.mul(self.network_params[i], self.tau) + tf.mul(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]
    
        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate, beta1 = MOMENTUM, beta2 = MOMENTUM_2).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action
        self.action_grads = tf.gradients(self.out, self.action)
        self.action_param_grads = tf.gradients(self.out, self.action_params)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.state_size])
        action_input = tflearn.input_data(shape=[None, self.action_dim])
        action_params = tflearn.input_data(shape=[None, self.action_param_dim])
        merged_input = tflearn.merge([inputs, action_input, action_params], 'concat', 1)
        w_init = tflearn.initializations.normal(stddev = 0.01)
        net = tflearn.fully_connected(merged_input, 1024, weights_init=w_init)
        net = tflearn.activation(tflearn.activations.leaky_relu(net, 0.01))
        
        w_init = tflearn.initializations.normal(stddev = 0.01)
        net = tflearn.fully_connected(net, 512, weights_init = w_init)
        net = tflearn.activation(tflearn.activations.leaky_relu(net, 0.01))
        
        w_init = tflearn.initializations.normal(stddev = 0.01)
        net = tflearn.fully_connected(net, 256, weights_init = w_init)
        net = tflearn.activation(tflearn.activations.leaky_relu(net, 0.01))
        
        w_init = tflearn.initializations.normal(stddev = 0.01)
        net = tflearn.fully_connected(net, 128, weights_init = w_init)
        net = tflearn.activation(tflearn.activations.leaky_relu(net, 0.01))

        # linear layer connected to 1 output representing Q(s,a) 
        # Weights are init to normal(0, 0.01)
        w_init = tflearn.initializations.normal(stddev=0.01)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action_input, action_params, out

    def train(self, inputs, action, action_params, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.action_params: action_params,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action, action_params):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.action_params: action_params
        })

    def predict_target(self, inputs, action, action_params):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action,
            self.target_action_params: action_params
        })

    def action_gradients(self, inputs, actions): 
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def action_param_gradients(self, inputs, action_params):
		return self.sess.run(self.action_params_grads, feed_dict={
            self.inputs: inputs,
            self.action_params: action_params
        })
        
    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries(): 
    episode_reward = tf.Variable(0.)
    tf.scalar_summary("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.scalar_summary("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.merge_all_summaries()

    return summary_ops, summary_vars

	
# ===========================
#   Agent Training
# ===========================
def get_param_offset(action, arg):
	if arg < 0 or arg > 1:
		return -1
	if action == 0:
		return arg
	if action == 1:
		if arg == 0:
			return 2
		return -1
	if action == 2:
		if arg == 0:
			return 3
		return -1
	return 4 + arg

def get_action(actor_output):
    actor_output[0,2] = -99999
    action = np.argmax(actor_output[0,0:ACTION_SIZE])
    arg1_offset = get_param_offset(action, 0)
    assert arg1_offset >= 0
    action_arg1 = actor_output[0,ACTION_SIZE + arg1_offset]
    arg2_offset = get_param_offset(action, 1)
    action_arg2 = 0
    if arg2_offset < 0:
        action_arg2 = 0
    else:
        action_arg2 = actor_output[0,ACTION_SIZE + arg2_offset]
    return action, arg1_offset, arg2_offset

def update(sess, replay_buffer, actor, critic):
    if len(replay_buffer) < MEMORY_THRESHOLD:
		return
    s_batch, a_batch, r_batch, t_batch, s2_batch = \
        replay_buffer.sample_batch(MINIBATCH_SIZE)

    # Calculate targets
    target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

    y_i = []
    for k in xrange(MINIBATCH_SIZE):
        if np.array_equal(s2_batch[k], np.zeros(actor.state_size)):
            y_i.append(r_batch[k])
        else:
            off_policy_target = r_batch[k] + GAMMA * target_q[k]
            on_policy_target = t_batch[k]
            assert np.isfinite(BETA * (r_batch[k] + GAMMA * target_q[k]) + (1 - BETA) * on_policy_target)
            y_i.append(BETA * (r_batch[k] + GAMMA * target_q[k]) + (1 - BETA) * on_policy_target)

		# Update the critic given the targets
        predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))
        assert(np.isfinite(predicted_q_value))
        critic.iterations += 1

		# Update the actor policy using the sampled gradient
        action_out, action_param_out = actor.predict(s_batch)                
        grads_action = critic.action_gradients(s_batch, action_out)
        grads_action_params = critic.action_param_gradients(s_batch, action_param_out)
        for n in xrange(MINIBATCH_SIZE):
            for h in xrange(ACTION_SIZE):
                maximum = 1.0
                mininum = -1.0
                if grads_action[n, h] < 0:
                    grads_action[n, h] *= (maximum - action_out[n][h]) / (maximum - mininum)
                elif grads_action[n, h] > 0:
                    grads_action[n, h] *= (action_out[n][h] - mininum) / (maximum - mininum)
            for h in xrange(PARAM_SIZE):
                maximum = 0
                mininum = 0
                if h == 0 or h == 4:
                    maximum = 100
                    mininum = 0
                elif h == 1 or h == 2 or h == 3 or h == 5:
                    maximum = 180
                    mininum = -180
                if grads_action_params[n, h] < 0:
                    grads_action_params[n, h] *= (maximum - action_param_out[n][h]) / (maximum - mininum)
                elif grads_action_params[n, h] > 0:
                    grads_action_params[n, h] *= (action_param_out[n][h] - mininum) / (maximum - mininum)
		actor.train(s_batch, grads_action, grads_action_params)
		actor.iterations += 1

		# Update target networks
		actor.update_target_network()
		critic.update_target_network()

	
def main(_):
    with tf.Session() as sess:
        
        env = gym.make(ENV_NAME)
        # np.random.seed(RANDOM_SEED)
        # tf.set_random_seed(RANDOM_SEED)
        # env.seed(RANDOM_SEED)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        assert (env.action_space.high == -env.action_space.low)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound, \
            ACTOR_LEARNING_RATE, TAU)

        critic = CriticNetwork(sess, state_dim, action_dim, \
            CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())

        if GYM_MONITOR_EN:
            if not RENDER_ENV:
                env.monitor.start(MONITOR_DIR, video_callable=False, force=True)
            else:
                env.monitor.start(MONITOR_DIR, force=True)

        train(sess, env, actor, critic)

        if GYM_MONITOR_EN:
            env.monitor.close()

if __name__ == '__main__':
    tf.app.run()
