import numpy as np
import tensorflow as tf
from replay_buffer import *
import tflearn
from params import *

class ActorNetwork():
    def __init__(self, sess, lr, tau, state_dim, action_dim, action_param_dim):
        self.sess = sess
        self.lr = lr
        self.tau = tau
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, GAMMA)
        self.iter = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_param_dim = action_param_dim

        # Create Actor Network
        self.inputs, self.action_output, self.action_param_output = self.create()
        self.network_params = tf.trainable_variables()

        self.inputs_target, self.action_output_target, self.action_param_output_target = self.create()
        self.network_params_target = tf.trainable_variables()[len(self.network_params):]

        self.action_gradient = tf.placeholder(tf.float32, [None, self.action_dim])
        self.action_param_gradient = tf.placeholder(tf.float32, [None, self.action_param_dim])

        self.opt = tf.train.AdamOptimizer(self.lr)
        self.actor_gradients = tf.gradients(self.action_output, self.network_params, grad_ys=self.action_gradient)
        self.actor_param_gradients = tf.gradients(self.action_param_output, self.network_params, grad_ys=self.action_param_gradient)

        grad_action = [(self.actor_gradients[i], self.network_params[i]) for i in xrange(len(self.network_params))]
        self.opt_action = self.opt.apply_gradients(grad_action)

        grad_param_action = [(self.actor_param_gradients[i], self.network_params[i]) for i in xrange(len(self.network_params))]
        self.opt_action_param = self.opt.apply_gradients(grad_param_action)


    def create(self):
        def leaky_relu(x):
            return tflearn.leaky_relu(x, alpha=0.01)

        inputs = tf.placeholder(tf.float32, shape=[None, self.state_dim])
        w_init = tf.random_normal_initializer(stddev=0.01)

        layer1 = tf.contrib.layers.fully_connected(inputs=inputs, \
            num_outputs=1024, activation_fn=leaky_relu, weights_initializer=w_init)

        layer2 = tf.contrib.layers.fully_connected(inputs=layer1, \
            num_outputs=512, activation_fn=leaky_relu, weights_initializer=w_init)

        layer3 = tf.contrib.layers.fully_connected(inputs=layer2, \
            num_outputs=256, activation_fn=leaky_relu, weights_initializer=w_init)

        layer4 = tf.contrib.layers.fully_connected(inputs=layer3, \
            num_outputs=128, activation_fn=leaky_relu, weights_initializer=w_init)

        action_output = tf.contrib.layers.fully_connected(inputs=layer4, \
            num_outputs=self.action_dim, weights_initializer=w_init)

        action_param_output = tf.contrib.layers.fully_connected(inputs=layer4, \
            num_outputs=self.action_param_dim, weights_initializer=w_init)

        return inputs, action_output, action_param_output


    def update_target_network_params(self):
        self.sess.run(\
            [self.network_params_target[i].assign(tf.mul(self.network_params[i], self.tau) + \
            tf.mul(self.network_params_target[i], 1 - self.tau)) \
            for i in xrange(len(self.network_params_target))])


    def train(self, inputs, action_gradient, action_param_gradient):
        self.sess.run([self.opt_action, self.opt_action_param], \
            feed_dict={ self.inputs: inputs,
                        self.action_gradient: action_gradient,
                        self.action_param_gradient: action_param_gradient })


    def predict(self, inputs):
        return self.sess.run([self.action_output, self.action_param_output],
            feed_dict={ self.inputs: inputs
        })


    def predict_target(self, inputs):
        return self.sess.run([self.action_output_target, self.action_param_output_target],
            feed_dict={ self.inputs_target: inputs
        })


    def action_selection(self, states, epsilon):
        if np.random.random() < epsilon:
            # With prob epsilon, return random action
            states = [states]
            action = []
            random_action = np.zeros((len(states), ACTION_SIZE + PARAM_SIZE))
            for i in xrange(len(states)):
                for j in xrange(ACTION_SIZE):
                    random_action[i, j] = np.random.uniform(-1., 1.)
                random_action[i, ACTION_SIZE] = np.random.uniform(-100. , 100.)
                random_action[i, ACTION_SIZE+1] = np.random.uniform(-180. , 180.)
                random_action[i, ACTION_SIZE+2] = np.random.uniform(-180. , 180.)
                random_action[i, ACTION_SIZE+3] = np.random.uniform(-180. , 180.)
                random_action[i, ACTION_SIZE+4] = np.random.uniform(0. , 100.)
                random_action[i, ACTION_SIZE+5] = np.random.uniform(-180. , 180.)
                action.append(random_action)
            action = np.array(action)[0]
        else:
            # Otherwise predict the action to return using the Actor Network
            inputs = [states]
            [action_output, action_param_output] = self.predict(inputs)
            action = np.hstack((action_output, action_param_output))

        return action


    def get_num_params(self):
        return len(self.network_params) + len(self.network_params_target)


class CriticNetwork():
    def __init__(self, sess, lr, tau, state_dim, action_dim, action_param_dim, num_actor_params):
        self.sess = sess
        self.lr = lr
        self.tau = tau
        self.iter = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_param_dim = action_param_dim

        self.inputs, self.actions, self.output = self.create()
        self.network_params = tf.trainable_variables()[num_actor_params:]

        self.inputs_target, self.actions_target, self.output_target = self.create()
        self.network_params_target = tf.trainable_variables()[len(self.network_params)+num_actor_params:]

        self.predicted_q_val = tf.placeholder(tf.float32, [None, 1])
        self.loss = tf.reduce_mean(tf.square(tf.sub(self.predicted_q_val, self.output)))
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


    def create(self):
        def leaky_relu(x):
            return tflearn.leaky_relu(x, alpha=0.01)

        states = tf.placeholder(tf.float32, [None, self.state_dim])
        actions = tf.placeholder(tf.float32, [None, self.action_dim + self.action_param_dim])
        inputs_merged = tf.concat(1, [states, actions])

        w_init = tf.random_normal_initializer(stddev=0.01)

        layer1 = tf.contrib.layers.fully_connected(inputs=inputs_merged, \
            num_outputs=1024, activation_fn=leaky_relu, weights_initializer=w_init)

        layer2 = tf.contrib.layers.fully_connected(inputs=layer1, \
            num_outputs=512, activation_fn=leaky_relu, weights_initializer=w_init)

        layer3 = tf.contrib.layers.fully_connected(inputs=layer2, \
            num_outputs=256, activation_fn=leaky_relu, weights_initializer=w_init)

        layer4 = tf.contrib.layers.fully_connected(inputs=layer3, \
            num_outputs=128, activation_fn=leaky_relu, weights_initializer=w_init)

        output = tf.contrib.layers.fully_connected(inputs=layer4, \
            num_outputs=1, weights_initializer=w_init)

        return states, actions, output


    def actor_gradients(self, inputs, actions):
        return self.sess.run(tf.gradients(self.output, self.actions), \
            feed_dict={ self.inputs: inputs,
                        self.actions: actions })


    def update_target_network_params(self):
        self.sess.run(\
            [self.network_params_target[i].assign(tf.mul(self.network_params[i], self.tau) + \
            tf.mul(self.network_params_target[i], 1 - self.tau)) \
            for i in xrange(len(self.network_params_target))])


    def train(self, inputs, actions, predicted_q_val):
        return self.sess.run([self.output, self.opt], \
            feed_dict={ self.inputs: inputs,
                        self.actions: actions,
                        self.predicted_q_val: predicted_q_val})


    def predict(self, inputs, actions):
        return self.sess.run(self.output, \
            feed_dict={ self.inputs: inputs,
                        self.actions: actions })


    def predict_target(self, inputs, actions):
        return self.sess.run(self.output_target, \
            feed_dict={ self.inputs_target: inputs,
                        self.actions_target: actions })


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
