import os
import pickle
import tensorflow as tf

from sklearn.metrics import f1_score

slim = tf.contrib.slim


class Model(object):
    def __init__(self, session, fp, lp):
        self.epoch = lp.EPOCH
        self.leaning_rate = lp.LEANING_RATE
        self.fold_size = lp.FOLD_SIZE

        self.data_path = fp.DATA_PATH
        self.save_path = fp.SAVE_PATH

        self.best_loss = 999999999.
        self.best_accuracy = 0.0

        self.session = session

        self._init_placeholder()
        self.model_5()

        self.optimizer()
        self.get_accuracy()

        tf.set_random_seed(410)

    def _init_placeholder(self):
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, 100, 3])
        self.target = tf.placeholder(dtype=tf.int64, shape=[None])
        self.keep_prob = tf.placeholder(dtype=tf.float32)

    def k_fold_validation(self, input_data, target_data):
        if len(input_data) != len(target_data):
            raise ValueError()

        data_size = len(input_data) // self.fold_size
        inputs, targets = [], []

        for i in range(self.fold_size - 1):
            inputs.append(input_data[i * data_size:(i + 1) * data_size])
            targets.append(target_data[i * data_size:(i + 1) * data_size])
        else:
            inputs.append(input_data[(self.fold_size - 1) * data_size:])
            targets.append(target_data[(self.fold_size - 1) * data_size:])

        return inputs, targets

    def model_1(self, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse) as encoder_scope:
            with slim.arg_scope([slim.conv2d], stride=(2, 1), padding='SAME', activation_fn=tf.nn.elu):
                x = slim.batch_norm(tf.reshape(slim.batch_norm(self.input), shape=[-1, 100, 3, 1]))
                x = slim.conv2d(x, 16, [5, 3], scope='Model_1')
                x = slim.conv2d(x, 32, [3, 3], scope='Model_2')
                x = slim.conv2d(x, 32, [3, 3], scope='Model_3')
                x = slim.conv2d(x, 32, [1, 1], stride=1, scope='Model_4')
                x = slim.conv2d(x, 64, [3, 3], scope='Model_5')
                x = slim.conv2d(x, 64, [3, 3], stride=2, scope='Model_6')
                x = slim.conv2d(x, 64, [1, 1], stride=1, scope='Model_7')
                x = slim.avg_pool2d(x, [4, 2])
                x = slim.flatten(x)

                self.x = slim.dropout(slim.fully_connected(x, 6), self.keep_prob)

    def model_2(self, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse) as encoder_scope:
            with slim.arg_scope([slim.conv2d], stride=(2, 1), padding='SAME', activation_fn=tf.nn.elu):
                data = slim.batch_norm(tf.reshape(slim.batch_norm(self.input), shape=[-1, 100, 3, 1]))
                branch_0 = slim.conv2d(data, 8, [5, 3], scope='Branch0_conv_1')
                branch_0 = slim.conv2d(branch_0, 16, [5, 3], scope='Branch0_conv_2')
                branch_0 = slim.conv2d(branch_0, 16, [1, 1], stride=1, scope='Branch0_conv_3')
                branch_0 = slim.conv2d(branch_0, 32, [5, 3], scope='Branch0_conv_4')
                branch_0 = slim.conv2d(branch_0, 32, [5, 3], stride=2, scope='Branch0_conv_5')
                branch_0 = slim.conv2d(branch_0, 32, [1, 1], stride=1, scope='Branch0_conv_6')
                branch_0 = slim.avg_pool2d(branch_0, [7, 2], scope='Branch0_avgpool_1')

                branch_1 = slim.conv2d(data, 8, [3, 3], scope='Branch1_conv_1')
                branch_1 = slim.conv2d(branch_1, 16, [3, 3], scope='Branch1_conv_2')
                branch_1 = slim.conv2d(branch_1, 16, [1, 1], stride=1, scope='Branch1_conv_3')
                branch_1 = slim.conv2d(branch_1, 32, [3, 3], scope='Branch1_conv_4')
                branch_1 = slim.conv2d(branch_1, 32, [3, 3], stride=2, scope='Branch1_conv_5')
                branch_1 = slim.conv2d(branch_1, 32, [1, 1], stride=1, scope='Branch1_conv_6')
                branch_1 = slim.avg_pool2d(branch_1, [7, 2], scope='Branch1_avgpool_1')

                x = slim.flatten(tf.concat(values=[branch_0, branch_1], axis=3))
                self.x = slim.dropout(slim.fully_connected(x, 6), self.keep_prob)

    def model_3(self, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse) as encoder_scope:
            with slim.arg_scope([slim.conv2d], stride=(2, 1), padding='SAME', activation_fn=tf.nn.elu):
                data = slim.batch_norm(tf.reshape(slim.batch_norm(self.input), shape=[-1, 100, 3, 1]))
                branch_0 = slim.conv2d(data, 8, [5, 3], scope='Branch0_conv_1')
                branch_0 = slim.conv2d(branch_0, 16, [5, 3], scope='Branch0_conv_2')
                branch_0 = slim.conv2d(branch_0, 16, [1, 1], stride=1, scope='Branch0_conv_3')
                branch_0 = slim.conv2d(branch_0, 32, [5, 3], scope='Branch0_conv_4')

                branch_1 = slim.conv2d(data, 8, [3, 3], scope='Branch1_conv_1')
                branch_1 = slim.conv2d(branch_1, 16, [3, 3], scope='Branch1_conv_2')
                branch_1 = slim.conv2d(branch_1, 16, [1, 1], stride=1, scope='Branch1_conv_3')
                branch_1 = slim.conv2d(branch_1, 32, [3, 3], scope='Branch1_conv_4')

                _x = tf.concat(values=[branch_0, branch_1], axis=3)

                branch_2 = slim.conv2d(_x, 64, [5, 3], stride=2, scope='Branch2_conv_1')
                branch_2 = slim.conv2d(branch_2, 64, [1, 1], stride=1, scope='Branch2_conv_2')
                branch_2 = slim.avg_pool2d(branch_2, [7, 2], scope='Branch2_avgpool_1')

                branch_3 = slim.conv2d(_x, 64, [3, 3], stride=2, scope='Branch3_conv_1')
                branch_3 = slim.conv2d(branch_3, 64, [1, 1], stride=1, scope='Branch3_conv_2')
                branch_3 = slim.avg_pool2d(branch_3, [7, 2], scope='Branch3_avgpool_1')

                x = slim.flatten(tf.concat(values=[branch_2, branch_3], axis=3))
                self.x = slim.dropout(slim.fully_connected(x, 6), self.keep_prob)

    def model_4(self, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse) as encoder_scope:
            with slim.arg_scope([slim.conv2d], stride=(2, 1), padding='SAME', activation_fn=tf.nn.elu):
                data = slim.batch_norm(tf.reshape(slim.batch_norm(self.input), shape=[-1, 100, 3, 1]))
                branch_0_0 = slim.conv2d(data, 8, [5, 3], scope='Branch_0_0_conv_1')
                branch_0_0 = slim.conv2d(branch_0_0, 16, [5, 3], scope='Branch_0_0_conv_2')
                branch_0_0 = slim.conv2d(branch_0_0, 16, [1, 1], stride=1, scope='Branch_0_0_conv_3')
                branch_0_0 = slim.conv2d(branch_0_0, 32, [5, 3], scope='Branch_0_0_conv_4')

                branch_0_1 = slim.conv2d(data, 8, [3, 3], scope='Branch_0_1_conv_1')
                branch_0_1 = slim.conv2d(branch_0_1, 16, [3, 3], scope='Branch_0_1_conv_2')
                branch_0_1 = slim.conv2d(branch_0_1, 16, [1, 1], stride=1, scope='Branch_0_1_conv_3')
                branch_0_1 = slim.conv2d(branch_0_1, 32, [3, 3], scope='Branch_0_1_conv_4')

                branch_0_2 = slim.conv2d(data, 8, [5, 3], scope='Branch_0_2_conv_1')
                branch_0_2 = slim.conv2d(branch_0_2, 16, [4, 3], scope='Branch_0_2_conv_2')
                branch_0_2 = slim.conv2d(branch_0_2, 16, [1, 1], stride=1, scope='Branch_0_2_conv_3')
                branch_0_2 = slim.conv2d(branch_0_2, 32, [3, 3], scope='Branch_0_2_conv_4')

                _x = tf.concat(values=[branch_0_0, branch_0_1, branch_0_2], axis=3)

                branch_1_0 = slim.conv2d(_x, 64, [5, 3], stride=2, scope='Branch_1_0_conv_1')
                branch_1_0 = slim.conv2d(branch_1_0, 64, [1, 1], stride=1, scope='Branch_1_0_conv_2')
                branch_1_0 = slim.avg_pool2d(branch_1_0, [7, 2], scope='Branch_1_0_avgpool_1')

                branch_1_1 = slim.conv2d(_x, 64, [3, 3], stride=2, scope='Branch_1_1_conv_1')
                branch_1_1 = slim.conv2d(branch_1_1, 64, [1, 1], stride=1, scope='Branch_1_1_conv_2')
                branch_1_1 = slim.avg_pool2d(branch_1_1, [7, 2], scope='Branch_1_1_avgpool_1')

                x = slim.flatten(tf.concat(values=[branch_1_0, branch_1_1], axis=3))
                self.x = slim.dropout(slim.fully_connected(x, 6), self.keep_prob)

    def model_5(self, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse) as encoder_scope:
            with slim.arg_scope([slim.conv2d], stride=(2, 1), padding='SAME', activation_fn=tf.nn.elu):
                data = slim.batch_norm(tf.reshape(slim.batch_norm(self.input), shape=[-1, 100, 3, 1]))
                branch_0_0 = slim.conv2d(data, 8, [5, 3], scope='Branch_0_0_conv_1')
                branch_0_0 = slim.conv2d(branch_0_0, 16, [5, 3], scope='Branch_0_0_conv_2')
                branch_0_0 = slim.conv2d(branch_0_0, 16, [1, 1], stride=1, scope='Branch_0_0_conv_3')
                branch_0_0 = slim.conv2d(branch_0_0, 32, [5, 3], scope='Branch_0_0_conv_4')

                branch_0_1 = slim.conv2d(data, 8, [3, 3], scope='Branch_0_1_conv_1')
                branch_0_1 = slim.conv2d(branch_0_1, 16, [3, 3], scope='Branch_0_1_conv_2')
                branch_0_1 = slim.conv2d(branch_0_1, 16, [1, 1], stride=1, scope='Branch_0_1_conv_3')
                branch_0_1 = slim.conv2d(branch_0_1, 32, [3, 3], scope='Branch_0_1_conv_4')

                branch_0_2 = slim.conv2d(data, 8, [5, 3], scope='Branch_0_2_conv_1')
                branch_0_2 = slim.conv2d(branch_0_2, 16, [4, 3], scope='Branch_0_2_conv_2')
                branch_0_2 = slim.conv2d(branch_0_2, 16, [1, 1], stride=1, scope='Branch_0_2_conv_3')
                branch_0_2 = slim.conv2d(branch_0_2, 32, [3, 3], scope='Branch_0_2_conv_4')

                _x = tf.concat(values=[branch_0_0, branch_0_1, branch_0_2], axis=3)

                branch_1_0 = slim.conv2d(_x, 48, [5, 3], stride=1, scope='Branch_1_0_conv_1')
                branch_1_0 = slim.conv2d(branch_1_0, 64, [1, 1], stride=1, scope='Branch_1_0_conv_2')
                branch_1_0 = slim.conv2d(branch_1_0, 64, [3, 3], stride=1, scope='Branch_1_0_conv_3')

                branch_1_1 = slim.conv2d(_x, 48, [3, 3], stride=1, scope='Branch_1_1_conv_1')
                branch_1_1 = slim.conv2d(branch_1_1, 64, [1, 1], stride=1, scope='Branch_1_1_conv_2')
                branch_1_1 = slim.conv2d(branch_1_1, 64, [3, 3], stride=1, scope='Branch_1_1_conv_3')

                mixed = tf.concat(values=[branch_1_0, branch_1_1], axis=3)
                up = slim.conv2d(mixed, _x.get_shape()[3], [1, 1], stride=1, padding='SAME', normalizer_fn=None,
                                 activation_fn=None, scope='Conv2d_up')

                scaled_up = up * 0.9
                _x += scaled_up
                _x = tf.nn.relu(_x)

                branch_2_0 = slim.conv2d(_x, 80, [5, 3], stride=2, scope='Branch_2_0_conv_1')
                branch_2_0 = slim.conv2d(branch_2_0, 80, [1, 1], stride=1, scope='Branch_2_0_conv_2')
                branch_2_0 = slim.avg_pool2d(branch_2_0, [7, 2], scope='Branch_2_0_avgpool_1')

                branch_2_1 = slim.conv2d(_x, 80, [3, 3], stride=2, scope='Branch_2_1_conv_1')
                branch_2_1 = slim.conv2d(branch_2_1, 80, [1, 1], stride=1, scope='Branch_2_1_conv_2')
                branch_2_1 = slim.avg_pool2d(branch_2_1, [7, 2], scope='Branch_2_1_avgpool_1')

                x = slim.flatten(tf.concat(values=[branch_2_0, branch_2_1], axis=3))
                self.x = slim.dropout(slim.fully_connected(x, 6), self.keep_prob)

    def optimizer(self):
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.x, labels=tf.one_hot(self.target, 6)))
        self.opt_loss = tf.train.AdamOptimizer(learning_rate=self.leaning_rate).minimize(self.loss)

    def get_accuracy(self):
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.x, 1), self.target), tf.float32))

    def test(self):
        data = pickle.load(open(os.path.join(self.data_path, 'test_set.pkl'), 'rb'))
        input_data, target = zip(*data)

        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(os.path.join(self.save_path, 'accuracy'))

        saver.restore(self.session, checkpoint.model_checkpoint_path)

        acc_1, t = self.session.run([self.accuracy, self.x],
                                  feed_dict={self.input: input_data, self.target: target, self.keep_prob: 1.0})
        print("\n!!! accuracy model's accuracy(test data) - accuracy: {}, f1_score: {} !!!".
              format(acc_1, f1_score(target, self.session.run(tf.argmax(t, 1)), average='micro')))
        ##########################################################################################################
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(os.path.join(self.save_path, 'loss'))

        saver.restore(self.session, checkpoint.model_checkpoint_path)

        acc_2, t = self.session.run([self.accuracy, self.x],
                               feed_dict={self.input: input_data, self.target: target, self.keep_prob: 1.0})
        print("!!! loss model's accuracy(test data) - accuracy: {}, f1_score: {} !!!".
              format(acc_2, f1_score(target, self.session.run(tf.argmax(t, 1)), average='micro')))

        return acc_1, acc_2

    def train(self):
        saver_loss = tf.train.Saver()
        saver_acc = tf.train.Saver()

        train_data = pickle.load(open(os.path.join(self.data_path, 'train_set.pkl'), 'rb'))
        input_data, target_data = zip(*train_data)
        inputs, targets = self.k_fold_validation(input_data, target_data)

        self.session.run(tf.global_variables_initializer())
        for ep in range(1, self.epoch + 1):
            loss_sum = 0.0
            for i in range(self.fold_size):
                if i == (ep % self.fold_size):
                    continue
                _, loss, _ = self.session.run([self.opt_loss, self.loss, self.x],
                                              feed_dict={self.input: inputs[i],
                                                         self.target: targets[i],
                                                         self.keep_prob: 0.9})
                loss_sum += loss

            acc = self.session.run(self.accuracy, feed_dict={self.input: inputs[ep % self.fold_size],
                                                             self.target: targets[ep % self.fold_size],
                                                             self.keep_prob: 0.9})

            if loss_sum < self.best_loss:
                self.best_loss = loss_sum
                print('model save(best loss model : epoch{})'.format(ep))
                saver_loss.save(self.session, os.path.join(self.save_path, 'loss', 'bestLoss'), global_step=ep)
            if acc > self.best_accuracy:
                self.best_accuracy = acc
                print('model save(best accuracy model : epoch{})'.format(ep))
                saver_acc.save(self.session, os.path.join(self.save_path, 'accuracy', 'bestAccuracy'), global_step=ep)

            if ep % 50 is 0:
                print('{}epoch average loss:'.format(ep), loss_sum)
                print('{}epoch validation accuracy:'.format(ep), acc)

        # print(self.best_accuracy, self.best_loss)
