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
    modelname = '20160808-005740'
    # modelname = config.modelname

    ckptpath = '/tmp/tfckpt/' + modelname + '/'
    ckpt = tf.train.get_checkpoint_state(ckptpath)
    print ckpt

    np.set_printoptions(precision=3, linewidth=150, suppress=True)

    datasets = get_datasets(config)

    print('Load Vocabs')
    # build vocab dict
    vocabfile = config.datafolder + 'vocab_10000.txt'
    vocab = get_list_from_file(vocabfile)
    rev_vocab = dict([(x, y) for (y, x) in enumerate(vocab)])

    # build short vocab dict
    vocabfile = config.datafolder + 'vocab_short_1000.txt'
    vocab_short = get_list_from_file(vocabfile)
    rev_vocab_short = dict([(x, y) for (y, x) in enumerate(vocab_short)])

    # build vocab dict
    vocabfile = config.datafolder + 'rel_vocab_8.txt'
    vocab_rel = get_list_from_file(vocabfile)
    rev_vocab_rel = dict([(x, y) for (y, x) in enumerate(vocab_rel)])

    print('BUILD GRAPH')
    m = RNNClassifierModel(config=config)

    data_batch = datasets.train.next_batch(1)
    feed_dict = make_feed_dict(m, data_batch, dropout_keep_prob=1.)

    saver = tf.train.Saver()

    with tf.Session() as sess:

        print('RESTORE VARIABLES FOR PREVIOUS MODEL')
        saver.restore(sess, ckpt.model_checkpoint_path)

        proba = sess.run(m.probas, feed_dict=feed_dict)
        print('probas:')
        print(proba)

        # probashort = sess.run(m.probas_short, feed_dict=feed_dict)
        # print('probashort:')
        # print(probashort)

        topk_short = sess.run(m.topk_short, feed_dict=feed_dict)

        top1 = [topk[0][1] for topk in topk_short]
        top1sent = [vocab_short[wrd] for wrd in top1]
        print top1sent

        # TODO: pull back top 3 with probs; map to vocabs.  print long, short actual, short top 3 etc.


if __name__ == "__main__":
    main()
