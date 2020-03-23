import tensorflow as tf
from tensorflow.contrib.layers import flatten, conv2d, fully_connected
import numpy as np


class DQN:
    def __init__(self, session, X, X_action, output_size, name="main", is_training=True):
        self.session = session

        self.output_size = output_size

        self._X = X

        self.net_name = name

        self.X_action = X_action

        if is_training:
            self.keep_prob = 0.7
        else:
            self.keep_prob = 1.0

        self._build_network()
        self.saver = tf.train.Saver()
        self.save_path = "./save/break/save_model_" + self.net_name + ".ckpt"
        # self.save_path2 = "./save/break/model_dqn"
        tf.logging.info(name + " - initialized")

    def _build_network(self, l_rate=0.3):
        with tf.variable_scope(self.net_name) as scope:
            keep_prob = self.keep_prob

            initializer = tf.contrib.layers.variance_scaling_initializer()

            # input place holders
            # X = tf.placeholder(tf.float32, shape=self.input_size)
            self.in_training_mode = tf.placeholder(tf.bool)
            self._keep_prob = tf.placeholder(tf.float32, name="kp")
            self.X_img = tf.reshape(self._X, [-1, 90, 90, 1])

            # Conv
            layer_1 = conv2d(self._X, num_outputs=32, kernel_size=(8, 8), stride=4, padding='SAME',
                             weights_initializer=initializer)
            tf.summary.histogram('layer_1', layer_1)

            layer_2 = conv2d(layer_1, num_outputs=64, kernel_size=(4, 4), stride=2, padding='SAME',
                             weights_initializer=initializer)
            tf.summary.histogram('layer_2', layer_2)

            layer_3 = conv2d(layer_2, num_outputs=64, kernel_size=(3, 3), stride=1, padding='SAME',
                             weights_initializer=initializer)
            tf.summary.histogram('layer_3', layer_3)

            # Flatten the result of layer_3 before feeding to the fully connected layer
            flat = flatten(layer_3)

            fc = fully_connected(flat, num_outputs=128, weights_initializer=initializer)
            tf.summary.histogram('fc', fc)

            self.output = fully_connected(fc, num_outputs=self.output_size, activation_fn=None,
                                          weights_initializer=initializer)
            tf.summary.histogram('output', self.output)

            self.vars = {v.name[len(scope.name):]: v for v in
                         tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)}

    def save(self, episode=0):
        self.saver.save(self.session, self.save_path + "-" + str(episode))

    def restore(self, episode=0):
        load_path = self.save_path + "-" + str(episode)
        # load_path2 = self.save_path2 + "-" + str(episode)
        print(load_path)
        self.saver.restore(self.session, load_path)

    def get_Q_action(self):
        return tf.reduce_sum(self.output * tf.one_hot(self.X_action, self.output_size), axis=-1, keep_dims=True)

    def get_actions(self, obs, in_training_mode=False, next_act=False):
        obs_final = [obs] if not next_act else obs
        return self.output.eval(feed_dict={self._X: obs_final, self.in_training_mode: False})
