import tensorflow as tf
import numpy as np
import pandas as pd
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

    config = cf.MixConfig(6)
    modelname = config.modelname

    ckptpath = '/data/tfckpt/' + modelname + '/'
    ckpt = tf.train.get_checkpoint_state(ckptpath)
    print ckpt

    np.set_printoptions(precision=3, linewidth=150, suppress=True, threshold=np.inf)

    datasets = get_datasets(config)

    print('Load Vocabs')
    # build relation vocab dict
    vocabfile = config.datafolder + 'rel_vocab_' + str(config.rel_vocab_size) +'.txt'
    vocab_rel = get_list_from_file(vocabfile)
    rev_vocab_rel = dict([(x, y) for (y, x) in enumerate(vocab_rel)])

    print('BUILD GRAPH')
    m = RNNClassifierModel(config=config)

    saver = tf.train.Saver()

    with tf.Session() as sess:

        print('RESTORE VARIABLES FOR PREVIOUS MODEL')
        saver.restore(sess, ckpt.model_checkpoint_path)

        feed_dict_test = make_feed_dict(m, datasets.test.data, dropout_keep_prob=1.)

        # Test accuracy
        stats_test = sess.run([m.actual_byclass, m.pred_byclass, m.correct_byclass,
                               m.precision_byclass, m.recall_byclass, m.f1_byclass],
                              feed_dict=feed_dict_test)
        actual, predict, correct, precision, recall, f1 = stats_test

        df = pd.DataFrame()
        df['relation'] = vocab_rel
        df['actual'] = actual
        df['predict'] = predict
        df['correct'] = correct
        df['precision'] = precision
        df['recall'] = recall
        df['f1'] = f1

        # move _unk relation to the end
        df = pd.concat([df.ix[1:], df.ix[[0]]])

        # calculate cumulative stats - 'micro average'
        # [note tensorflow 0.10 includes tf.cumsum - could use this in model.py when upgrade to 0.10]
        df['actual_cum'] = df.actual.cumsum()
        df['predict_cum'] = df.predict.cumsum()
        df['correct_cum'] = df.correct.cumsum()
        df['precision_cum'] = df.correct_cum / df.predict_cum
        df['recall_cum'] = df.correct_cum / df.actual_cum
        df['f1_cum'] = 2. * (df.precision_cum * df.recall_cum) / (df.precision_cum + df.recall_cum)

        # calculate unweighted averages over classes = 'macro average'
        # classcounts = range(1, actual.size+1)
        # precision_cumsum = np.cumsum(results_df['precision'])
        # recall_cumsum = np.cumsum(results_df['recall'])
        # f1_cumsum = np.cumsum(results_df['f1'])
        # precision_cumavg = np.true_divide(precision_cumsum, classcounts)
        # recall_cumavg = np.true_divide(recall_cumsum, classcounts)
        # f1_cumavg = np.true_divide(f1_cumsum, classcounts)

        df['precision_macroavg'] = df.precision.expanding().mean()
        df['recall_macroavg'] = df.recall.expanding().mean()
        df['f1_macroaavg'] = df.f1.expanding().mean()

        print df
        df.to_csv('/data/eval/' + modelname + '.csv')


if __name__ == "__main__":
    main()
