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
import copy

from replay_buffer import ReplayBuffer

# ==========================
#   Training Parameters
# ==========================
MOMENTUM = 0.95
MOMENTUM_2 = 0.999
MAX_NORM = 10
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.00001
# ACTOR_LEARNING_RATE = 0.001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor 
GAMMA = 0.99
# Soft target update param
TAU = 0.001
BETA = 0.5
STATE_INPUT_COUNT = 1
# Size of replay buffer
BUFFER_SIZE = 500000
MEMORY_THRESHOLD = 1000
MINIBATCH_SIZE = 32
ACTION_SIZE = 4


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
    def __init__(self, sess, state_size, action_dim, tid, learning_rate, tau):
        self.sess = sess
        self.state_size = state_size
        self.state_input_data_size = MINIBATCH_SIZE * state_size
        self.action_dim = action_dim
        self.tid = tid
        self.learning_rate = learning_rate
        self.tau = tau
        self.unum = 0
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, GAMMA)
        self.iterations = 0

        # Actor Network
        layer_sizes = [self.state_size, 1024, 1]
        weights_initial = self.actor_initial_weights(layer_sizes)
        self.inputs, self.dash_turn = self.create_actor_network(weights_initial, layer_sizes)
        
        self.dashturn_network_params = tf.trainable_variables()
        
        self.target_inputs, self.target_dash_turn = self.create_actor_network(weights_initial, layer_sizes)
        
        self.target_dashturn_network_params = tf.trainable_variables()[len(self.dashturn_network_params):]
        for i in xrange(len(self.target_dashturn_network_params)):
            self.target_dashturn_network_params[i].assign(self.dashturn_network_params[i])
        
        self.update_dashturn_target_network_params = \
            [self.target_dashturn_network_params[i].assign(tf.mul(self.dashturn_network_params[i], self.tau) + \
                tf.mul(self.target_dashturn_network_params[i], 1. - self.tau))
                for i in range(len(self.target_dashturn_network_params))]
        
        self.all_dashturn_network_params = tf.trainable_variables()
        
        layer_sizes = [self.state_size, 1024, self.action_dim - 1]
        weights_initial = self.actor_initial_weights(layer_sizes)
        self.inputs_2, self.action_out = self.create_actor_network(weights_initial, layer_sizes)
        
        self.network_params = tf.trainable_variables()
        # Target Network
        self.target_inputs_2, self.target_action_out = self.create_actor_network(weights_initial, layer_sizes)
        
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]
        for i in xrange(len(self.target_network_params)):
            self.target_network_params[i].assign(self.network_params[len(self.all_dashturn_network_params) + i])
        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.mul(self.network_params[len(self.all_dashturn_network_params) + i], self.tau) + \
                tf.mul(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]
        
        # This gradient will be provided by the critic network
        self.dashturn_grad = tf.placeholder(tf.float32, [None, 1])
        self.action_gradient = tf.placeholder(tf.float32, [None, self.action_dim - 1])
        
        # Combine the gradients here 
        assert self.action_out != None
        assert self.network_params != None
        assert self.action_gradient != None
        '''
        self.networks_params = \
            [tf.concat(1, [self.dashturn_network_params[i], self.network_params[len(self.all_dashturn_network_params) + i]])
                for i in range(len(self.dashturn_network_params))]
        
        print self.networks_params
        '''
        self.actor_gradients = tf.gradients(self.action_out, self.network_params[len(self.all_dashturn_network_params):], -self.action_gradient)
        self.dashturn_gradients = tf.gradients(self.dash_turn, self.dashturn_network_params, -self.dashturn_grad)
        
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1 = MOMENTUM, beta2 = MOMENTUM_2)
        # self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        assert self.actor_gradients != None
        self.optimize = self.optimizer.apply_gradients(zip(self.actor_gradients, self.network_params[len(self.all_dashturn_network_params):]))
        self.optimize_dashturn = self.optimizer.apply_gradients(zip(self.dashturn_gradients, self.dashturn_network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params) + len(self.dashturn_network_params) + len(self.target_dashturn_network_params)
    
    def actor_initial_weights(self, layer_sizes):
        weights_initial = []
        for i in xrange(len(layer_sizes) - 1):
            w_init = tflearn.initializations.normal(stddev = 0.01)
            '''
            with tf.Session():
                weights_initial.append(tf.Variable(w_init.eval()))
            '''
            weights_initial.append(w_init)
        return weights_initial
    
    def create_actor_network(self, weights_initial, layer_sizes): 
        inputs = tflearn.input_data(shape=[None, self.state_size])
        
        net = tflearn.fully_connected(inputs, layer_sizes[1], weights_init=weights_initial[0])
        net = tflearn.activation(tflearn.activations.leaky_relu(net, 0.01))
 
        for i in xrange(2, len(layer_sizes) - 1):
            net = tflearn.fully_connected(net, layer_sizes[i], weights_init=weights_initial[i - 1])
            net = tflearn.activation(tflearn.activations.leaky_relu(net, 0.01))
        
        # Final layer weights are init to Normal(0, 0.01)
        action_out = tflearn.fully_connected(net, layer_sizes[len(layer_sizes) - 1], weights_init = weights_initial[len(weights_initial) - 1])
        return inputs, action_out

    def train(self, inputs, action_gradient, dashturn_gradient):
        self.sess.run([self.optimize_dashturn, self.optimize], feed_dict={
            self.inputs: inputs,
            self.inputs_2: inputs,
            self.action_gradient: action_gradient,
            self.dashturn_grad: dashturn_gradient
        })

    def predict(self, inputs):
        predicted = self.sess.run([self.dash_turn, self.action_out], feed_dict={
            self.inputs: inputs,
            self.inputs_2: inputs
        })
        return np.concatenate((predicted[0], predicted[1]), 1)
    
    def predict_target(self, inputs):
        predicted = self.sess.run([self.target_dash_turn, self.target_action_out], feed_dict={
            self.target_inputs: inputs,
            self.target_inputs_2: inputs
        })
        # print predicted[0].shape, predicted[1].shape
        return np.concatenate((predicted[0], predicted[1]), 1)

    def update_target_network(self):
        self.sess.run(self.update_dashturn_target_network_params)
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
    
    def get_random_actor_output(self):
        random_action = np.zeros(self.action_dim)
        random_action[0] = np.random.uniform(0, 1)
        random_action[1] = np.random.uniform(-100.0, 100.0)
        random_action[2] = np.random.uniform(-180.0, 180.0)
        random_action[3] = np.random.uniform(-180.0, 180.0)
        return random_action

    def select_action(self, state, epsilon):
        return (self.select_actions([state], epsilon))[0]

    def select_actions(self, states, epsilon):
        assert epsilon >= 0.0 and epsilon <= 1.0
        assert len(states) <= MINIBATCH_SIZE
        if np.random.uniform() < epsilon:
            actor_output = np.zeros((len(states), self.action_dim))
            for i in xrange(len(actor_output)):
                actor_output[i] = self.get_random_actor_output()
            # print actor_output
            return actor_output
        # print actor_output
        return self.predict(states)

class CriticNetwork(object):
    """ 
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """
    def __init__(self, sess, state_size, action_dim, tid, learning_rate, tau, num_actor_vars):
        self.sess = sess
        self.state_size = state_size
        self.action_dim = action_dim
        self.tid = tid
        self.learning_rate = learning_rate
        self.tau = tau
        self.unum = 0
        self.iterations = 0

        # Create the critic network
        layer_sizes = [self.state_size + self.action_dim, 1024, 1]
        weights_initial = self.critic_initial_weights(layer_sizes)
        self.inputs, self.action, self.out = self.create_critic_network(weights_initial, layer_sizes)

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network(weights_initial, layer_sizes)
        
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]
        for i in xrange(len(self.target_network_params)):
            self.target_network_params[i].assign(self.network_params[i])
        
        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.mul(self.network_params[i], self.tau) + tf.mul(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]
    
        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate, beta1 = MOMENTUM, beta2 = MOMENTUM_2).minimize(self.loss)
        # self.optimize = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action
        self.action_grads = tf.gradients(self.out, self.action)

    def critic_initial_weights(self, layer_sizes):
        weights_initial = []
        for i in xrange(len(layer_sizes) - 1):
            w_init = tflearn.initializations.normal(stddev = 0.01)
            weights_initial.append(w_init)
        return weights_initial
    
    def create_critic_network(self, weights_initial, layer_sizes):
        inputs = tflearn.input_data(shape=[None, self.state_size])
        action = tflearn.input_data(shape=[None, self.action_dim])
        net = tflearn.merge([inputs, action], 'concat', 1)
        for i in xrange(1, len(layer_sizes) - 1):
            net = tflearn.fully_connected(net, layer_sizes[i], weights_init=weights_initial[i - 1])
            net = tflearn.activation(tflearn.activations.leaky_relu(net, 0.01))
        # linear layer connected to 1 output representing Q(s,a) 
        # Weights are init to normal(0, 0.01)
        out = tflearn.fully_connected(net, 1, weights_init=weights_initial[len(weights_initial) - 1])
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict = {
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, action): 
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

def get_action(actor_output):
    if actor_output[0] <= 0.5:
        return 0, actor_output[1], actor_output[2]
    else:
        return 1, actor_output[3], 0
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

def update(sess, replay_buffer, actor, critic):
    if replay_buffer.size() < MEMORY_THRESHOLD:
        return
    s_batch, a_batch, r_batch, t_batch, s2_batch = \
        replay_buffer.sample_batch(MINIBATCH_SIZE)

    # Calculate targets
    target_actor_action = actor.predict_target(s2_batch)
    target_q = critic.predict_target(s2_batch, target_actor_action)

    y_i = []
    for k in xrange(MINIBATCH_SIZE):
        off_policy_target = 0
        if np.array_equal(s2_batch[k], np.zeros(actor.state_size)):
            off_policy_target = r_batch[k]
        else:
            off_policy_target = r_batch[k] + GAMMA * target_q[k]
        on_policy_target = t_batch[k]
        target = BETA * on_policy_target + (1 - BETA) * off_policy_target
        assert np.isfinite(target)
        y_i.append(target)

    predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))
    # print predicted_q_value.shape
    # print predicted_q_value
    # assert(np.isfinite(critic_loss))
    critic.iterations += 1

    # Update the actor policy using the sampled gradient
    action_out = actor.predict(s_batch)
    grads_action = critic.action_gradients(s_batch, action_out)[0]
    for n in xrange(MINIBATCH_SIZE):
        for h in xrange(ACTION_SIZE):
            if h == 0:
                maximum = 1.0
                mininum = 0.0
            elif h == 1:
                maximum = 100.0
                mininum = -100.0
            else:
                maximum = 180.0
                minimum = -180.0
            if grads_action[n, h] > 0:
                grads_action[n, h] *= (maximum - action_out[n][h]) / (maximum - mininum)
            elif grads_action[n, h] < 0:
                grads_action[n, h] *= (action_out[n][h] - mininum) / (maximum - mininum)
    # print grads_action[:,0].shape
    actor.train(s_batch, grads_action[:,1:], np.reshape(grads_action[:,0], (MINIBATCH_SIZE, 1)))
    actor.iterations += 1

    # Update target networks
    actor.update_target_network()
    critic.update_target_network()

    return np.mean(predicted_q_value)
	
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
