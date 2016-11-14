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
        print '[ddpg] reward=[%f]'%(reward)
        new_state = game.get_state()
        new_state_size = game.get_state_size()

        if game.status == hfo_py.IN_GAME:
            transition = (current_state, action_selected, reward, 0, new_state)
            episode_buffer.append(transition)

    # Relabel the experiences
    actor.replay_buffer.label(episode_buffer)

    # Store the tried experiences in the replay buffer
    actor.replay_buffer.addMultiple(episode_buffer)

    return game.total_reward, game.steps, game.status

def run():
    #num_players = OFFENSE_AGENTS + OFFENSE_NPCS + DEFENSE_AGENTS + \
    #    DEFENSE_NPCS + OFFENSE_DUMMIES + DEFENSE_DUMMIES + DEFENSE_CHASERS
    #state_dim = get_state_dim(num_players-1)

    with tf.Session() as sess:

        print '[ddpg] Establishing Game Environment ...'
        game = GymSoccerState(0)
        game.create_env()
        state_dim = game.get_state_size()

        print '[ddpg] Creating actor / critic network ...'
        actor = ActorNetwork(sess, ACTOR_LEARNING_RATE, TAU, state_dim, ACTION_SIZE, PARAM_SIZE, SIMPLE)
        critic = CriticNetwork(sess, CRITIC_LEARNING_RATE, TAU, state_dim, ACTION_SIZE, PARAM_SIZE, actor.get_num_params(), SIMPLE)

        summary_ops, summary_vars = build_summaries()
        writer = tf.train.SummaryWriter(SUMMARY_DIR, sess.graph)

        print '[ddpg] Initializing variables ...'
        sess.run(tf.initialize_all_variables())
        num_iter = max(actor.iter, critic.iter)

        saver = tf.train.Saver()

        print '[ddpg] Running iterations ...'
        while num_iter < MAX_ITER:
            if game.episode_over:
                game.reset(0)
            num_iter = max(actor.iter, critic.iter)
            #if num_iter % 1000 == 0:
            #    print '=========iter %d========'%num_iter
            #print actor.replay_buffer.size()
            epsilon = anneal_epsilon(num_iter)
            (total_reward, steps, status) = playSingleEpisode(game, actor, critic, epsilon)
            print '[ddpg] Episode Summary -----'
            print '  epsilon: [%f]'%(epsilon)
            print '  total_reward:[%f]'%(total_reward)
            print '----------------------------'
            #print '\t[ddpg] total_reward:[%f], status:[%s]'%(total_reward, status)
            #print '[ddpg] Updating actor / critic networks ...'

            # Start learning once buffer has more than MEMORY_THRESHOLD examples.
            if actor.replay_buffer.size() < MEMORY_THRESHOLD:
                continue

            #print len(actor.network_params), len(actor.network_params_target)
            predicted_q_val, loss = actor_critic(actor, critic)
            qmax = np.max(predicted_q_val)
            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: total_reward,
                summary_vars[1]: qmax,
                summary_vars[2]: loss
            })

            writer.add_summary(summary_str, num_iter)
            writer.flush()

            if num_iter % 1000 == 0:
                save_path = saver.save(sess, "%s/model_%d.ckpt"%(SUMMARY_DIR, num_iter))

if __name__ == '__main__':
    run()
