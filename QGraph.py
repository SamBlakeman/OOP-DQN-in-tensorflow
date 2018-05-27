import os
import tensorflow as tf
import numpy as np


class QGraph(object):
    def __init__(self, im_width, im_height, m, num_actions, directory):
        self.ti = 0
        self.num_actions = num_actions
        self.directory = directory

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        # tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            """ Construction phase """
            self.X = tf.placeholder(tf.float32, shape=(None, im_width, im_height, m), name="X")
            self.y = tf.placeholder(tf.float32, shape=(None), name="y")
            self.actions = tf.placeholder(tf.float32, shape=[None, self.num_actions], name="actions")

            # Layers
            self.conv1 = tf.layers.conv2d(inputs=self.X, filters=32, kernel_size=[8, 8], strides=4, padding="same",
                                          activation=tf.nn.relu)
            self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64, kernel_size=[4, 4], strides=2, padding="same",
                                          activation=tf.nn.relu)
            self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=64, kernel_size=[3, 3], strides=1, padding="same",
                                          activation=tf.nn.relu)
            self.conv3_flat = tf.reshape(self.conv3, [-1, 11 * 11 * 64])
            self.dense = tf.layers.dense(inputs=self.conv3_flat, units=512, activation=tf.nn.relu)
            self.logits = tf.layers.dense(inputs=self.dense, units=self.num_actions)

            # Loss function
            with tf.name_scope("loss"):
                self.predictions = tf.reduce_sum(tf.multiply(self.logits, self.actions), 1)
                self.targets = tf.stop_gradient(self.y)


                self.error = self.targets - self.predictions
                self.clipped_error = tf.clip_by_value(self.targets - self.predictions, -1., 1.)
                self.loss = tf.reduce_mean(tf.multiply(self.error, self.clipped_error), axis=0, name='loss')

            # Minimizer
            self.learning_rate = 0.00025
            self.momentum = 0.95
            self.epsilon = 0.01
            self.batch_size = 32

            with tf.name_scope("train"):
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, momentum=self.momentum, epsilon=self.epsilon)
                self.training_op = self.optimizer.minimize(self.loss)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

            tf.add_to_collection('logits', self.logits)

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

        return

    def GetActionValues(self, X):
        preds = self.logits.eval(feed_dict={self.X: np.divide(X, 255)}, session=self.sess)

        return preds

    def GradientDescentStep(self, X_batch, action_batch, y_batch):
        # One hot encoded action tensor
        actions = np.zeros((self.batch_size, self.num_actions))
        for i in range(self.batch_size):
            actions[i, action_batch[i]] = 1

        self.sess.run(self.training_op,
                      feed_dict={self.X: np.divide(X_batch, 255), self.y: y_batch, self.actions: actions})

        return

    def SaveGraphAndVariables(self):
        save_path = self.saver.save(self.sess, self.directory)
        print('Model saved in ' + save_path)

        return

    def LoadGraphAndVariables(self):
        self.saver.restore(self.sess, self.directory)
        print('Model loaded from ' + self.directory)

        return







