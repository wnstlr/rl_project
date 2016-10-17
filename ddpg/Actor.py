import tensorflow as tf

class Actor():
    def __init__(self, sess, lr, tau, action_bound):
        self.sess = sess
        self.lr = lr
        self.tau = tau
        self.action_bound = action_bound

        self.params = tf.trainable_variables()
        self.target_params = tf.trainable_variables()

        ## Define layer information
        self.inputs = tf.placeholder(tf.float32, shape=(None,))
        self.hidden1 = tf.nn.relu(inputs)

        ## Optimizer
        opt = tf.train.AdamOptimizer(self.lr)


    def train(self, inputs, action_grad):
        self.sess.run(self.optimize, feed_dict={self.inputs: inputs, \
                                                self.action_grad = action_grad})



