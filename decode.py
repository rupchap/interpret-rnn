import tensorflow as tf
import numpy as np
from load_data import get_datasets, get_list_from_file
import time
import pickle

from model import RNNClassifierModel
from train import Config, make_feed_dict
from datasets import split_data

# To run tensorboard:
# tensorboard --logdir=/tmp/tflogs


def main():

    config = Config()

    ckptpath = '/tmp/tfckpt/' + config.modelname + '/'
    ckptfile = ckptpath + 'model.ckpt'
    ckpt = tf.train.get_checkpoint_state(ckptpath)
    print ckpt

    datasets = get_datasets(config)

    # build vocab dict
    vocabfile = config.datafolder + 'vocab_10000.txt'
    vocab = dict([(x, y) for (y, x) in enumerate(get_list_from_file(vocabfile))])

    # build vocab dict
    vocabfile = config.datafolder + 'rel_vocab_8.txt'
    vocab_rel = dict([(x, y) for (y, x) in enumerate(get_list_from_file(vocabfile))])


    print('BUILD GRAPH')
    m = RNNClassifierModel(config=config)  # , init_embedding=init_embedding)

    data_batch = datasets.train.next_batch(2)
    feed_dict = make_feed_dict(m, data_batch, dropout_keep_prob=1.)

    saver = tf.train.Saver()

    with tf.Session() as sess:

        print('RESTORE VARIABLES FOR PREVIOUS MODEL')
        saver.restore(sess, ckpt.model_checkpoint_path)

        proba = sess.run(m.proba, feed_dict=feed_dict)
        print('probas:')
        print(proba)


        probashort = sess.run(m.short_probas, feed_dict=feed_dict)
        print('probashort:')
        print(probashort)

#         TODO: pull back top 3 with probs; map to vocabs.  print long, short actual, short top 3 etc.


if __name__ == "__main__":
    main()
