from train import *
from gym_soccer_interface import *

def playSingleEpisode(game, actor, critic, epsilon):
    episode_buffer = []
    while not game.episode_over:
        current_state = game.get_state()
        current_state_size = game.get_state_size()
        action_selected = actor.action_selection(current_state, epsilon)[0]
        action, param1, param2 = output2action(action_selected)
        print '[ddpg] %s'%action2string(action, param1, param2)
        new_states, rewards, episode_over, _ = game.act(action, param1, param2)
        game.update()
        reward = game.reward()
        new_state = game.get_state()
        new_state_size = game.get_state_size()

        if game.status == hfo_py.IN_GAME:
            transition = (current_state, action_selected, reward, 0, new_state)
            episode_buffer.append(transition)

    # Store the tried experiences in the replay buffer
    actor.replay_buffer.addMultiple(episode_buffer)

    return game.total_reward, game.steps, game.status

def run():
    num_players = OFFENSE_AGENTS + OFFENSE_NPCS + DEFENSE_AGENTS + \
        DEFENSE_NPCS + OFFENSE_DUMMIES + DEFENSE_DUMMIES + DEFENSE_CHASERS
    state_dim = get_state_dim(num_players)

    with tf.Session() as sess:
        print '[ddpg] Establishing Game Environment ...'
        game = GymSoccerAgainstKeeperState(0)
        game.create_env()

        print '[ddpg] Creating actor / critic network ...'
        actor = ActorNetwork(sess, ACTOR_LEARNING_RATE, TAU, state_dim, ACTION_SIZE, PARAM_SIZE)
        critic = CriticNetwork(sess, CRITIC_LEARNING_RATE, TAU, state_dim, ACTION_SIZE, PARAM_SIZE, actor.get_num_params())

        print '[ddpg] Initializing variables ...'
        sess.run(tf.initialize_all_variables())
        num_iter = max(actor.iter, critic.iter)

        print '[ddpg] Running iterations ...'

        # epsilon = anneal_epsilon(num_iter)
        # (total_reward, steps, status) = playSingleEpisode(game, actor, critic, epsilon)
        # actor_critic(actor, critic)

        while num_iter < MAX_ITER:
            if game.episode_over:
                game.reset(0)
            num_iter = max(actor.iter, critic.iter)
            #if num_iter % 1000 == 0:
            #    print '=========iter %d========'%num_iter
            #print actor.replay_buffer.size()
            epsilon = anneal_epsilon(num_iter)
            (total_reward, steps, status) = playSingleEpisode(game, actor, critic, epsilon)
            #print '\t[ddpg] total_reward:[%f], status:[%s]'%(total_reward, status)
            #print '[ddpg] Updating actor / critic networks ...'
            actor_critic(actor, critic)

if __name__ == '__main__':
    run()
