import tensorflow as tf
import numpy as np
from load_data import get_datasets
from load_data import build_initial_embedding
import time
import os

from model import RNNClassifierModel

# To run tensorboard:
# tensorboard --logdir=/tmp/tflogs


class Config(object):
    # init_scale = 0.1
    learning_rate = 1.
    lr_decay = 0.99
    max_grad_norm = 5

    embed_size = 200    # 50, 100, 200 or 300 to match glove embeddings
    hidden_size = 150

    max_sentence_length = 106
    vocab_size = 10000

    max_shortsentence_length = 15
    vocab_size_short = 1000

    rel_vocab_size = 12

    dropout_keep_prob = .5

    train_size = 0  # 0 to use all remaining data for training.
    validation_size = 1000
    test_size = 0

    batch_size = 64

    report_step = 200
    save_step = 2000

    srcfile = '/data/NYT/nyt-freebase.train.triples.universal.mention.txt'
    datafolder = '/data/train/'
    embedfolder = '/data/glove/'

    cost_with_relation = True
    cost_with_short = False

    # model name - if provided, will seek to load previous checkpoint and continue training.
    modelname = ''


def main():

    np.set_printoptions(precision=3, linewidth=150, suppress=True)

    config = Config()

    if config.modelname:
        modelname = config.modelname
    else:
        modelname = time.strftime("%Y%m%d-%H%M%S")

    logfolder = '/data/tflogs/' + modelname + '/'
    ckptpath = '/data/tfckpt/' + modelname + '/'
    if not os.path.exists(ckptpath):
        os.makedirs(ckptpath)
    ckptfile = ckptpath + 'model.ckpt'

    # Create a saver for writing checkpoints.
    ckpt = tf.train.get_checkpoint_state(ckptpath)
    print ckpt

    print('LOAD DATA')
    datasets = get_datasets(config)
    print('%i training examples loaded' % datasets.train.num_examples)
    print('batch size = %i  1 epoch = %i steps' % (config.batch_size, datasets.train.num_examples / config.batch_size))

    print('BUILD GRAPH')
    m = RNNClassifierModel(config=config)

    # Op to generate summary stats
    merged = tf.merge_all_summaries()

    saver = tf.train.Saver()

    print('RUN TRAINING')

    with tf.Session() as sess:

        if ckpt:
            print('RESTORE VARIABLES FOR MODEL: %s' % modelname)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('INITIALISE VARIABLES FOR NEW MODEL: %s' % modelname)
            init_op = tf.initialize_all_variables()
            sess.run(init_op)
            print('APPLY INITIAL WORD EMBEDDINGS')
            init_embedding = build_initial_embedding(config)
            sess.run(m.embedding.assign(init_embedding))

        # instantiate SummaryWriters to output summaries and the Graph.
        writer = tf.train.SummaryWriter(logfolder + 'train/', graph=sess.graph)
        writer_val = tf.train.SummaryWriter(logfolder + 'val/')

        # get full training data feed_dict - no dropout
        data_train = datasets.train.data
        feed_dict_train = make_feed_dict(m, data_train, dropout_keep_prob=1.0)

        # get validation data feed_dict - no dropout
        data_val = datasets.validation.data
        feed_dict_val = make_feed_dict(m, data_val, dropout_keep_prob=1.0)

        # keep track of validation costs to adjust learning rate when needed
        previous_val_costs = []

        # loop over training steps
        while True:

            # train on a batch of training data
            data_batch = datasets.train.next_batch(config.batch_size)
            feed_dict = make_feed_dict(m, data_batch, config.dropout_keep_prob)
            _, global_step, cost_batch, summaries = sess.run([m.train_op, m.global_step, m.cost, merged],
                                                             feed_dict=feed_dict)

            writer.add_summary(summaries, global_step=global_step)

            # Report stats and consider reducing LR
            if global_step > 0 and global_step % config.report_step == 0:
                summaries_val, cost_val = sess.run([merged, m.cost], feed_dict=feed_dict_val)
                writer_val.add_summary(summaries, global_step=global_step)
                print('step:%2i batch cost:%8f validation cost:%8f' % (global_step, cost_batch, cost_val))

                # Training accuracy
                stats_trn = sess.run([m.actual_byclass, m.pred_byclass, m.precision_byclass,
                                      m.recall_byclass, m.f1_byclass],
                                     feed_dict=feed_dict_train)
                print('training - actuals/predictions/precision/recall/f1:')
                print(np.stack(stats_trn))

                # Validation accuracy
                stats_val = sess.run([m.actual_byclass, m.pred_byclass, m.precision_byclass,
                                      m.recall_byclass, m.f1_byclass],
                                     feed_dict=feed_dict_val)
                print('validation - actuals/predictions/precision/recall/f1:')
                print(np.stack(stats_val))

                # decay learning rate if not improving
                if len(previous_val_costs) > 3 and cost_val > max(previous_val_costs[-4:]):
                    m.decay_lr(sess, config.lr_decay)
                    print('decayed lr to:', sess.run(m.lr, feed_dict))
                previous_val_costs.append(cost_val)

            if global_step > 0 and global_step % config.save_step == 0:
                print('Saving model to: %s' % ckptfile)
                saver.save(sess, ckptfile)

    print 'Optimisation complete'


def make_feed_dict(m, data, dropout_keep_prob=1.):
    feed_dict = {m.input_data: data['sentence'],
                 m.lengths: data['sentence_lengths'],
                 m.input_data_short: data['short'],
                 m.lengths_short: data['short_lengths'],
                 m.weights_short: data['short_weights'],
                 m.targets: data['relation'],
                 m.dropout_keep_prob: dropout_keep_prob}
    return feed_dict


if __name__ == "__main__":
    main()
