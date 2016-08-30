import tensorflow as tf
import numpy as np
from load_data import get_datasets, get_list_from_file
import time
import pickle

from model import RNNClassifierModel
from train import make_feed_dict
import configs as cf
from datasets import split_data

# To run tensorboard:
# tensorboard --logdir=/tmp/tflogs


def main():

    config = cf.DefaultConfig()
    modelname = config.modelname

    ckptpath = '/data/tfckpt/' + modelname + '/'
    ckpt = tf.train.get_checkpoint_state(ckptpath)
    print ckpt

    np.set_printoptions(precision=3, linewidth=150, suppress=True)

    datasets = get_datasets(config)

    print('Load Vocabs')
    # build vocab dict
    vocabfile = config.datafolder + 'vocab_' + str(config.vocab_size) + '.txt'
    vocab = get_list_from_file(vocabfile)
    rev_vocab = dict([(x, y) for (y, x) in enumerate(vocab)])

    # build short vocab dict
    vocabfile = config.datafolder + 'vocab_short_' + str(config.vocab_size_short) + '.txt'
    vocab_short = get_list_from_file(vocabfile)
    rev_vocab_short = dict([(x, y) for (y, x) in enumerate(vocab_short)])

    # build vocab dict
    vocabfile = config.datafolder + 'rel_vocab_' + str(config.rel_vocab_size) +'.txt'
    vocab_rel = get_list_from_file(vocabfile)
    rev_vocab_rel = dict([(x, y) for (y, x) in enumerate(vocab_rel)])

    print('BUILD GRAPH')
    m = RNNClassifierModel(config=config)

    saver = tf.train.Saver()

    with tf.Session() as sess:

        print('RESTORE VARIABLES FOR PREVIOUS MODEL')
        saver.restore(sess, ckpt.model_checkpoint_path)

        while True:
            data_batch = datasets.validation.next_batch(1, include_text=True)

            print('long original: ' + data_batch['sentence_text'][0])
            print('entA original: ' + data_batch['enta_text'][0])
            print('entB original: ' + data_batch['entb_text'][0])
            print('short original: ' + data_batch['short_text'][0])
            print('relation original: ' + data_batch['relation_text'][0])

            longsentence = ' '.join([vocab[data_batch['sentence'][0][i]] for i in range(data_batch['sentence_lengths'][0])])
            shortsentence = ' '.join([vocab_short[data_batch['short'][0][i]] for i in range(data_batch['short_lengths'][0])])
            relation = vocab_rel[data_batch['relation'][0]]

            print('long processed: ' + longsentence)
            print('short processed: ' + shortsentence)
            print('relation processed: ' + relation)

            feed_dict = make_feed_dict(m, data_batch, dropout_keep_prob=1.)
            sht, rel = sess.run([m.topk_short, m.topk_rel], feed_dict=feed_dict)

            top1 = [topk[0] for topk in sht]
            top1sent = ' '.join([vocab_short[wrd] for wrd in top1])
            relation_pred = vocab_rel[rel[0][0]]
            print('predicted short: ' + top1sent)
            print('predicted relation: ' + relation_pred)

            raw_input("Press Enter to continue...")
            print('---------------------------------------------------------------------------------')


if __name__ == "__main__":
    main()
