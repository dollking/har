import os
import time
import tensorflow as tf
from tensorflow.python.framework import ops

from model.model_test import Model
from preprocessing.preprocessing import PreProcessing

from config import FilePath as fp
from config import LeaningParameter as lp
from config import PreProcessingParameter as pp


def analyzer(data_list):
    data_list.sort()
    return min(data_list), max(data_list), data_list[len(data_list) // 2], sum(data_list) / len(data_list)


if __name__ == '__main__':
    preprocess = PreProcessing(fp, pp)
    preprocess.preprocess()

    acc_model_list, loss_model_list = [], []
    for _ in range(5):
        sess = tf.Session()
        model = Model(sess, fp, lp)
        model.train()

        acc1, acc2 = model.test()
        acc_model_list.append(acc1)
        loss_model_list.append(acc2)

        os.system('rm save/accuracy/*')
        os.system('rm save/loss/*')

        time.sleep(1)
        ops.reset_default_graph()
        sess.close()
        time.sleep(1)

    analyzed_data = analyzer(acc_model_list)
    print('accuracy model - min: {},  max: {},  median: {},  average: {}'.format(analyzed_data[0], analyzed_data[1],
                                                                              analyzed_data[2], analyzed_data[3]))
    analyzed_data = analyzer(loss_model_list)
    print('loss model - min: {},  max: {},  median: {},  average: {}'.format(analyzed_data[0], analyzed_data[1],
                                                                          analyzed_data[2], analyzed_data[3]))
