from params import *
from network import *
import copy

# ====================
# Actor-Critic updates
# ====================
def actor_critic(actor, critic, grad_type='invert'):
    # Start learning once buffer has more than MEMORY_THRESHOLD examples.
    if actor.replay_buffer.size() < MEMORY_THRESHOLD:
        return

    # Sample a random minibatch of MINIBATCH_SIZE transitions from buffer
    s_batch, a_batch, r_batch, t_batch, s2_batch = \
        actor.replay_buffer.sample_batch(MINIBATCH_SIZE)

    # Set y_i
    target_action, target_action_params = actor.predict_target(s2_batch)
    target_actions = np.hstack((target_action, target_action_params))
    target_q_val = critic.predict_target(s2_batch, target_actions)

    #y_i = []
    #for i in xrange(MINIBATCH_SIZE):
    #    y_i.append(r_batch[k] + GAMMA * target_q_val[k])
    y_i = r_batch + GAMMA * target_q_val

    # Update critic by minimizing the loss
    predicted_q_val, opt = critic.train(s_batch, a_batch, y_i)
    critic.iter += 1

    # Update the actor policy using the sampled policy gradient
    action_output, action_param_output = actor.predict(s_batch)
    action_grads_combined = critic.actor_gradients(s_batch, np.hstack((action_output, action_param_output)))
    action_grads = action_grads_combined[0][:,:ACTION_SIZE]
    action_param_grads = action_grads_combined[0][:,ACTION_SIZE:]

    if grad_type == 'regular':
        # No other operations for bounded parameters
        action_grads_out = action_grads
        action_param_grads_out = action_param_grads
    elif grad_type == 'invert':
        # Invert the gradient for bounded parameters
        action_grads_out, action_param_grads_out = \
            invert_gradient(action_grads, action_param_grads, action_output, action_param_output)
    elif grad_type == 'zero':
        # Zero the gradient for bounded parameters
        action_grads_out, action_param_grads_out = \
            zero_gradient(action_grads, action_param_grads, action_output, action_param_output)
    elif grad_type == 'squash':
        # Squash the gradient for bounded parameters
        action_grads_out, action_param_grads_out = \
            squash_gradient(action_grads, action_param_grads, action_output, action_param_output)
    else:
        raise ValueError('input grad_type [%s] not defined.'%grad_type)

    actor.train(s_batch, action_grads_out, action_param_grads_out)
    actor.iter += 1

    # Update the target networks
    critic.update_target_network_params()
    actor.update_target_network_params()

# =============================================
# Gradient Modifications for bounded parameters
# =============================================
def invert_gradient(action_grads, action_param_grads, action_output, action_param_output):
    action_grads_out = copy.deepcopy(action_grads)
    action_param_grads_out = copy.deepcopy(action_param_grads)

    for i in xrange(MINIBATCH_SIZE):
        pmax = 1.
        pmin = -1.
        for j in xrange(ACTION_SIZE):
            if action_grads[i, j] > 0:
                action_grads_out[i, j] *= (pmax - action_output[i, j]) / (pmax - pmin)
            else:
                action_grads_out[i, j] *= (action_output[i, j] - pmax) / (pmax - pmin)
        for k in xrange(PARAM_SIZE):
            if k == 0:
                pmax = 100.
                pmin = -100.
            elif k == 4:
                pmax = 100.
                pmin = 0.
            else:
                pmax = 180.
                pmin = -180.
            if action_param_grads[i, k] > 0:
                action_param_grads_out[i, k] *= (pmax - action_param_output[i, k]) / (pmax - pmin)
            else:
                action_param_grads_out[i, k] *= (action_param_output[i, k] - pmax) / (pmax - pmin)

    return action_grads_out, action_param_grads_out

def zero_gradient(action_grads, action_param_grads, action_output, action_param_output):
    action_grads_out = copy.deepcopy(action_grads)
    action_param_grads_out = copy.deepcopy(action_param_grads)

    for i in xrange(MINIBATCH_SIZE):
        pmax = 1.
        pmin = -1.
        for j in xrange(ACTION_SIZE):
            if action_grads[i, j] > pmax or action_grads[i, j] < pmin:
                action_grads_out[i, j] = 0
        for k in xrange(PARAM_SIZE):
            if k == 0:
                pmax = 100.
                pmin = -100.
            elif k == 4:
                pmax = 100.
                pmin = 0.
            else:
                pmax = 180.
                pmin = -180.
            if action_param_grads[i, k] > pmax or action_param_grads[i, k] < pmin:
                action_param_grads_out[i, k] = 0

    return action_grads_out, action_param_grads_out

def squash_gradient(action_grads, action_param_grads, action_output, action_param_output):
    raise NotImplementedError

# =============================
# Helper functions
# =============================
def get_state_dim(num_players):
    return 58 + 8 * num_players

def anneal_epsilon(iter):
    if iter < EXPLORE_ITER:
        return 1.0 - (1.0 - EPSILON) * (iter / EXPLORE_ITER)
    return EPSILON

# converts the actor output to semantic values for actions to use in gym soccer
def output2action(actor_output):
    actor_output[2] = -1000 ## we don't want to sample TACKLE
    action = np.argmax(actor_output[:ACTION_SIZE])
    p1, p2 = param_index(action)
    if p2 is None:
        param1 = actor_output[ACTION_SIZE+p1]
        param2 = None
    else:
        param1 = actor_output[ACTION_SIZE+p1]
        param2 = actor_output[ACTION_SIZE+p2]

    return action, param1, param2

# converts action and params int values to string
def action2string(action, param1, param2):
    if action == 0: #DASH
        res = 'action [%s] : [%0.2f, %0.2f]'%('DASH', param1, param2)
    elif action == 1: #TURN
        res = 'action [%s] : [%0.2f]'%('TURN', param1)
    elif action == 2: #TACKLE
        res = 'action [%s] : [%0.2f]'%('TACKLE', param1)
    elif action == 3: #KICK
        res = 'action [%s] : [%0.2f, %0.2f]'%('KICK', param1, param2)
    return res

# Gets the index offset of the params related to that action
def param_index(action):
    if action == 0:
        #DASH
        return 0, 1
    elif action == 1:
        #TURN
        return 2, None
    elif action == 2:
        # TACKLE
        return 3, None
    elif action == 3:
        # KICK
        return 4, 5
