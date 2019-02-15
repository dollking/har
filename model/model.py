import os
import random
import pickle
import tensorflow as tf

from config import LeaningParameter as lp
from config import FilePath as fp


slim = tf.contrib.slim


class Model(object):
    def __init__(self, session):
        self.epoch = lp.EPOCH
        self.leaning_rate = lp.LEANING_RATE
        self.batch_size = lp.BATCH_SIZE

        self.session = session

        self._init_placeholder()
        self.encoder()
        self.decoder()

        self.VAE_optimizer()

        tf.set_random_seed(410)

    def _init_placeholder(self):
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, 250, 3])
        self.target = tf.placeholder(dtype=tf.float32, shape=[None])

    def encoder(self, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse) as encoder_scope:
            with slim.arg_scope([slim.conv2d], stride=(2, 1), padding='SAME', activation_fn=tf.nn.elu):
                self.norm_data = slim.batch_norm(self.input)
                x = slim.batch_norm(tf.reshape(self.norm_data, shape=[-1, 250, 3, 1]))
                x = slim.conv2d(x, 16, [5, 1], scope='Encoder_1')
                x = slim.conv2d(x, 32, [3, 1], scope='Encoder_2')
                x = slim.conv2d(x, 32, [3, 3], scope='Encoder_3')
                x = slim.conv2d(x, 64, [3, 3], scope='Encoder_4')
                x = slim.conv2d(x, 64, [3, 3], scope='Encoder_5')
                x = slim.conv2d(x, 96, [3, 3], stride=1, scope='Encoder_6')
                x = slim.conv2d(x, 96, [3, 3], stride=2, scope='Encoder_7')
                x = slim.conv2d(x, 128, [3, 2], scope='Encoder_8')
                x = slim.conv2d(x, 128, [2, 2], stride=2, scope='Encoder_9')
                # self.encoded_value = slim.conv2d(x, 128, [2, 2], stride=2, scope='Encoder_9')

                x = tf.contrib.layers.flatten(x)
                self.mn = tf.layers.dense(x, units=128)
                self.sd = 0.5 * tf.layers.dense(x, units=128)
                epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], 128]))
                self.z = self.mn + tf.multiply(epsilon, tf.exp(self.sd))

    def decoder(self, reuse=False):
        with tf.variable_scope('decoder', reuse=reuse) as decoder_scope:
            x = tf.reshape(self.z, [-1, 1, 1, 128])
            x = tf.nn.conv2d_transpose(x, tf.Variable(tf.random_normal([1, 1, 128, 128])),
                                       [self.batch_size, 2, 2, 128], [1, 2, 2, 1], name='Decoder_1')
            x = tf.nn.conv2d_transpose(x, tf.Variable(tf.random_normal([2, 2, 96, 128])), [self.batch_size, 4, 2, 96],
                                       [1, 2, 1, 1], name='Decoder_2')
            x = tf.nn.conv2d_transpose(x, tf.Variable(tf.random_normal([3, 2, 96, 96])), [self.batch_size, 8, 3, 96],
                                       [1, 2, 2, 1], name='Decoder_3')
            x = tf.nn.conv2d_transpose(x, tf.Variable(tf.random_normal([3, 3, 64, 96])), [self.batch_size, 8, 3, 64],
                                       [1, 1, 1, 1], name='Decoder_4')
            x = tf.nn.conv2d_transpose(x, tf.Variable(tf.random_normal([3, 3, 64, 64])), [self.batch_size, 16, 3, 64],
                                       [1, 2, 1, 1], name='Decoder_5')
            x = tf.nn.conv2d_transpose(x, tf.Variable(tf.random_normal([3, 3, 32, 64])), [self.batch_size, 32, 3, 32],
                                       [1, 2, 1, 1], name='Decoder_6')
            x = tf.nn.conv2d_transpose(x, tf.Variable(tf.random_normal([3, 3, 32, 32])), [self.batch_size, 63, 3, 32],
                                       [1, 2, 1, 1], name='Decoder_7')
            x = tf.nn.conv2d_transpose(x, tf.Variable(tf.random_normal([3, 1, 16, 32])), [self.batch_size, 125, 3, 16],
                                       [1, 2, 1, 1], name='Decoder_8')
            self.decoded_value = tf.nn.conv2d_transpose(x, tf.Variable(tf.random_normal([5, 1, 1, 16])),
                                                        [self.batch_size, 250, 3, 1], [1, 2, 1, 1], name='Decoder_9')

    def VAE_optimizer(self):
        img_loss = tf.reduce_sum(tf.squared_difference(tf.reshape(self.decoded_value, [-1, 250, 3]), self.norm_data))
        latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * self.sd - tf.square(self.mn) - tf.exp(2.0 * self.sd), 1)
        self.VAE_loss = tf.reduce_mean(img_loss + latent_loss)
        self.opt_VAE_loss = tf.train.AdamOptimizer(learning_rate=self.leaning_rate).minimize(self.VAE_loss)

    def siamese_loss_op(self):
        pass

    def train(self):
        train_data = pickle.load(open(os.path.join(fp.DATA_PATH, 'train_set.pkl'), 'rb'))

        self.session.run(tf.global_variables_initializer())
        for ep in range(1, self.epoch + 1):
            random.shuffle(train_data)
            input_data, target = zip(*train_data)

            loss_avg = 0.0
            for i in range(0, len(input_data), 107):
                _, loss, _ = self.session.run([self.opt_VAE_loss, self.VAE_loss, self.decoded_value],
                                              feed_dict={self.input: input_data[i:i + 107], self.target: target[i:i + 107]})
                loss_avg += loss
            print('{}epoch average loss: '.format(ep), loss_avg / (len(input_data) // 107))

