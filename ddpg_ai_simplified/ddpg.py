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
        # layer_sizes = [self.state_size, 1024, 512, 256, 128, self.action_dim, self.action_param_dim]
        # layer_sizes = [self.state_size, 256, 128, self.action_dim, self.action_param_dim]
        layer_sizes = [self.state_size, 1024, self.action_dim, self.action_param_dim]
        weights_initial = self.actor_initial_weights(layer_sizes)
        self.inputs, self.action_out, self.action_param_out = self.create_actor_network(weights_initial, layer_sizes)

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_action_out, self.target_action_param_out = self.create_actor_network(weights_initial, layer_sizes)
        
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]
        for i in xrange(len(self.target_network_params)):
            self.target_network_params[i].assign(self.network_params[i])
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
        gradient_list = self.actor_gradients + self.actor_param_gradients
        clipped_grads, _ = tf.clip_by_global_norm(gradient_list, MAX_NORM)
        clipped_actor_grads = clipped_grads[0:len(self.actor_gradients)]
        clipped_actor_param_grads = clipped_grads[len(self.actor_gradients):]
        # Optimization Op
        # Not sure how to clip gradients to norm 10, I get a ValueException: None Values Not Supported error
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1 = MOMENTUM, beta2 = MOMENTUM_2)
        assert self.actor_gradients != None
        assert MAX_NORM != None
        self.optimize_action = self.optimizer.apply_gradients(zip(clipped_actor_grads, self.network_params))
        self.optimize_action_param = self.optimizer.apply_gradients(zip(clipped_actor_param_grads, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)
    
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
        
        net = tflearn.fully_connected(inputs, 1024, weights_init=weights_initial[0])
        net = tflearn.activation(tflearn.activations.leaky_relu(net, 0.01))
 
        for i in xrange(2, len(layer_sizes) - 2):
            net = tflearn.fully_connected(net, layer_sizes[i], weights_init=weights_initial[i - 1])
            net = tflearn.activation(tflearn.activations.leaky_relu(net, 0.01))
        
        # Final layer weights are init to Normal(0, 0.01)
        action_out = tflearn.fully_connected(net, self.action_dim, weights_init = weights_initial[len(weights_initial) - 2])
        action_param_out = tflearn.fully_connected(net, self.action_param_dim, weights_init = weights_initial[len(weights_initial) - 1])
        return inputs, action_out, action_param_out

    def train(self, inputs, a_gradient, a_param_gradient):
        '''
        a_combined_gradient = tf.concat(1, [a_gradient, a_param_gradient])
        a_clipped_gradient = tf.clip_by_norm(a_combined_gradient, MAX_NORM)
        action_clipped_gradient = a_clipped_gradient[:,0:ACTION_SIZE]
        action_param_clipped_gradient = a_clipped_gradient[:,ACTION_SIZE:ACTION_SIZE+PARAM_SIZE]
        with tf.Session():
            action_clipped_gradient = action_clipped_gradient.eval()
            action_param_clipped_gradient = action_param_clipped_gradient.eval()
        # print action_clipped_gradient
        # print action_param_clipped_gradient
        '''
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
        return (self.select_actions([state], epsilon))[0]

    def select_actions(self, states, epsilon):
        assert epsilon >= 0.0 and epsilon <= 1.0
        assert len(states) <= MINIBATCH_SIZE
        if np.random.uniform() < epsilon:
            actor_output = np.zeros((len(states), ACTION_SIZE + PARAM_SIZE))
            for i in xrange(len(actor_output)):
                actor_output[i] = self.get_random_actor_output()
            # print actor_output
            return actor_output
        action_list = self.predict(states)
        actor_output= np.zeros((len(states), ACTION_SIZE + PARAM_SIZE))
        actor_output = np.concatenate((action_list[0], action_list[1]), 1)
        # print actor_output
        return actor_output
			
    def sample_action(self, actor_output):
        dash_prob = max(0, actor_output[0] + 1.0)
        turn_prob = max(0, actor_output[1] + 1.0)
        """
        Set tackle probability to 0 since there is no use to tackling in 1 vs 0 situation. Also kick probability to 0 since we're just trying to get to ball.
        """
        tackle_prob = 0
        kick_prob = 0
        action_type = np.random.choice(4, p = [dash_prob, turn_prob, tackle_prob, kick_prob])
        arg1 = get_param_offset(action_type, 0)
        assert arg1 >= 0
        action_arg1 = actor_output[ACTION_SIZE + arg1]
        arg2 = self.get_param_offset(action_type, 1)
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
        #self.inputs, self.action, self.action_params, self.out = self.create_critic_network()
        #self.action_all = tflearn.merge([self.action, self.action_params], 'concat', 1)
        # layer_sizes = [self.state_size + self.action_dim + self.action_param_dim, 1024, 512, 256, 128, 1]
        # layer_sizes = [self.state_size + self.action_dim + self.action_param_dim, 256, 128, 1]
        layer_sizes = [self.state_size + self.action_dim + self.action_param_dim, 1024, 1]
        weights_initial = self.critic_initial_weights(layer_sizes)
        self.inputs, self.action_all, self.out = self.create_critic_network(weights_initial, layer_sizes)

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        #self.target_inputs, self.target_action, self.target_action_params, self.target_out = self.create_critic_network()
        self.target_inputs, self.target_action_all, self.target_out = self.create_critic_network(weights_initial, layer_sizes)
        
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
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1 = MOMENTUM, beta2 = MOMENTUM_2)
        '''
        self.optimize_compute = self.optimizer.compute_gradients(self.loss)
        '''
        # print tf.gradients(self.loss, self.network_params)
        '''
        self.optimize_compute = self.optimizer.compute_gradients(self.loss)
        self.computed_gradients_clipped = self.clip_gradients(self.optimize_compute)
        self.optimize_apply = self.optimizer.apply_gradients(self.computed_gradients_clipped)
        '''
        clipped_grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.network_params), MAX_NORM)
        self.optimize_apply = self.optimizer.apply_gradients(zip(clipped_grads, self.network_params))
        '''
        self.optimize_compute = self.optimizer.minimize(self.loss)
        '''
        # Get the gradient of the net w.r.t. the action
        #self.action_all_grads = tf.gradients(self.out, tflearn.merge([self.action, self.action_params], 'concat', 1))
        self.action_all_grads = tf.gradients(self.out, self.action_all)
        # self.action_param_grads = tf.gradients(self.out, self.action_params)

    def critic_initial_weights(self, layer_sizes):
        weights_initial = []
        for i in xrange(len(layer_sizes) - 1):
            w_init = tflearn.initializations.normal(stddev = 0.01)
            weights_initial.append(w_init)
            '''
            with tf.Session():
                weights_initial.append(tf.Variable(w_init.eval()))
            '''
        return weights_initial
    
    def create_critic_network(self, weights_initial, layer_sizes):
        inputs = tflearn.input_data(shape=[None, self.state_size])
        action_all = tflearn.input_data(shape=[None, self.action_dim + self.action_param_dim])
        #action_input = tflearn.input_data(shape=[None, self.action_dim])
        #action_params = tflearn.input_data(shape=[None, self.action_param_dim])
        # merged_input = tflearn.merge([inputs, action_input, action_params], 'concat', 1)
        net = tflearn.merge([inputs, action_all], 'concat', 1)
        for i in xrange(1, len(layer_sizes) - 1):
            net = tflearn.fully_connected(net, layer_sizes[i], weights_init=weights_initial[i - 1])
            net = tflearn.activation(tflearn.activations.leaky_relu(net, 0.01))
        '''
        net = tflearn.fully_connected(net, 1024, weights_init = weights_initial[0])
        net = tflearn.activation(tflearn.activations.leaky_relu(net, 0.01))
        
        net = tflearn.fully_connected(net, 512, weights_init = weights_initial[1])
        net = tflearn.activation(tflearn.activations.leaky_relu(net, 0.01))
        
        net = tflearn.fully_connected(net, 256, weights_init = weights_initial[2])
        net = tflearn.activation(tflearn.activations.leaky_relu(net, 0.01))
        
        net = tflearn.fully_connected(net, 128, weights_init = weights_initial[3])
        net = tflearn.activation(tflearn.activations.leaky_relu(net, 0.01))
        '''
        # linear layer connected to 1 output representing Q(s,a) 
        # Weights are init to normal(0, 0.01)
        out = tflearn.fully_connected(net, 1, weights_init=weights_initial[len(weights_initial) - 1])
        #return inputs, action_input, action_params, out
        return inputs, action_all, out

    '''
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

    def action_all_gradients(self, inputs, actions): 
        return self.sess.run(self.action_all_grads, feed_dict={
            self.inputs: inputs,
            self.action_all: actions
        })
    
    def clip_gradients(self, computed_gradients):
        clip_grads = True
        computed_gradients_clipped = computed_gradients[:]
        rows = 0
        for gv in computed_gradients:
            if gv[0] != None:
                rows = gv[0].get_shape().as_list()[0]
        for i in xrange(len(computed_gradients)):
            if computed_gradients[i][0] == None:
                computed_gradients_clipped[i] = (tf.zeros([rows, 1]), computed_gradients_clipped[i][1])
        if clip_grads:
    
    
            grads_combined = gv[0][0]
            for i in xrange(len(optimize_compute)):
                if i > 0:
                    grads_combined.concat(1, gv[i][0]
           
            
            grads_shapes = [gv[0].get_shape().as_list() for gv in computed_gradients_clipped]
            print computed_gradients_clipped
            grads_combined = tf.concat(1, [gv[0] for gv in computed_gradients_clipped])
            grads_clipped = tf.clip_by_norm(tf.Variable(grads_combined), MAX_NORM)
            for i in xrange(len(computed_gradients)):
                if grads_clipped[:,0:grads_shapes[i]] == 0:
                    computed_gradients_clipped.append((None, gv[i][1]))
                elif i == 0:
                    computed_gradients_clipped.append((grads_clipped[:,0:grads_shapes[i]], gv[i][1]))
                else:
                    computed_gradients_clipped.append((grads_clipped[:,grads_shapes[i-1]:grads_shapes[i]], gv[i][1]))
        return computed_gradients_clipped

    def replace_none_with_zero(l):
        return [0 if i==None else i for i in l] 
    '''
    def train(self, inputs, action_all, predicted_q_value):
        return self.sess.run([self.out, self.optimize_apply], feed_dict = {
            self.inputs: inputs,
            self.action_all: action_all,
            self.predicted_q_value: predicted_q_value
        })
        '''
        print output
        return output
        '''
        # print self.optimize_compute
        '''
        computed_gradients = tf.gradients(
        gradients_clipped = self.clip_gradients(self.optimize_compute)
        # print clipped_gradients
        feed_dict = {}
        print gradients_clipped
        for i, grad_var in enumerate(gradients_clipped):
            if gradients_clipped[i][0] == None:
                feed_dict[self.computed_gradients_clipped[i][1]] = None
            else:
                print gradients_clipped[i][0]
                feed_dict[self.computed_gradients_clipped[i][1]] = self.sess.run(gradients_clipped[i][0], feed_dict = {
                    self.inputs: inputs,
                    self.action_all: action_all,
                    self.predicted_q_value: predicted_q_value
                })
        print feed_dict
        applied_grads = self.sess.run(self.optimize_apply.run(feed_dict=feed_dict), feed_dict = {
                    self.inputs: inputs,
                    self.action_all: action_all,
                    self.predicted_q_value: predicted_q_value
        })
        # print applied_grads
        return [output, applied_grads]
        '''
        '''
        print self.sess.run([self.out, self.optimizer.minimize(self.loss)], feed_dict = {
            self.inputs: inputs,
            self.action_all: action_all,
            self.predicted_q_value: predicted_q_value
        })
        return self.sess.run([self.out, self.optimizer.minimize(self.loss)], feed_dict = {
            self.inputs: inputs,
            self.action_all: action_all,
            self.predicted_q_value: predicted_q_value
        })
        '''
        '''
        list_of_stuff = self.sess.run([self.out, self.optimize_apply], feed_dict={
            self.inputs: inputs,
            self.action_all: action_all,
            self.predicted_q_value: predicted_q_value
        })
        print list_of_stuff[1]
        return list_of_stuff
        '''
    def predict(self, inputs, action_all):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action_all: action_all
        })

    def predict_target(self, inputs, action_all):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action_all: action_all
        })

    def action_all_gradients(self, inputs, action_all): 
        return self.sess.run(self.action_all_grads, feed_dict={
            self.inputs: inputs,
            self.action_all: action_all
        })
    '''
    def action_param_gradients(self, inputs, action_params):
        return self.sess.run(self.action_params_grads, feed_dict={
            self.inputs: inputs,
            self.action_params: action_params
        })
    '''
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
    actor_output_copy = actor_output[:]
    # Disable tackling and kicking since we're just trying to get to ball
    # actor_output_copy[1] = -99999
    actor_output_copy[2] = -99999
    actor_output_copy[3] = -99999
    action = np.argmax(actor_output_copy[0:ACTION_SIZE])
    arg1_offset = get_param_offset(action, 0)
    assert arg1_offset >= 0
    action_arg1 = actor_output[ACTION_SIZE + arg1_offset]
    arg2_offset = get_param_offset(action, 1)
    action_arg2 = 0
    if arg2_offset < 0:
        action_arg2 = 0
    else:
        action_arg2 = actor_output[ACTION_SIZE + arg2_offset]
    # print action, action_arg1, action_arg2
    return action, action_arg1, action_arg2

def update(sess, replay_buffer, actor, critic):
    if replay_buffer.size() < MEMORY_THRESHOLD:
        return
    s_batch, a_batch, r_batch, t_batch, s2_batch = \
        replay_buffer.sample_batch(MINIBATCH_SIZE)

    # Calculate targets
    target_actor_action = actor.predict_target(s2_batch)
    target_action_all = np.concatenate((target_actor_action[0], target_actor_action[1]), 1)
    target_q = critic.predict_target(s2_batch, target_action_all)

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

    # Update the critic given the targets
    # print len(y_i), MINIBATCH_SIZE
    # print a_batch.shape
    # action_batch = a_batch[:,0:ACTION_SIZE]
    # action_params_batch = a_batch[:,ACTION_SIZE:ACTION_SIZE+PARAM_SIZE]
    predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))
    # print predicted_q_value.shape
    # print predicted_q_value
    # assert(np.isfinite(critic_loss))
    critic.iterations += 1

    # Update the actor policy using the sampled gradient
    action_out, action_param_out = actor.predict(s_batch)
    # print s_batch.shape, action_out.shape
    # print action_param_out                
    action_all = np.concatenate((action_out, action_param_out), 1)
    # print action_all.shape
    grads_action_all = critic.action_all_gradients(s_batch, action_all)
    # print grads_action_all
    grads_action = grads_action_all[0][:,0:ACTION_SIZE]
    #print grads_action
    # grads_action_params = critic.action_param_gradients(s_batch, action_param_out)
    grads_action_params = grads_action_all[0][:,ACTION_SIZE:ACTION_SIZE+PARAM_SIZE]
    for n in xrange(MINIBATCH_SIZE):
        for h in xrange(ACTION_SIZE):
            maximum = 1.0
            mininum = -1.0
            if grads_action[n, h] > 0:
                grads_action[n, h] *= (maximum - action_out[n][h]) / (maximum - mininum)
            elif grads_action[n, h] < 0:
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
            if grads_action_params[n, h] > 0:
                grads_action_params[n, h] *= (maximum - action_param_out[n][h]) / (maximum - mininum)
            elif grads_action_params[n, h] < 0:
                grads_action_params[n, h] *= (action_param_out[n][h] - mininum) / (maximum - mininum)
    actor.train(s_batch, grads_action, grads_action_params)
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
