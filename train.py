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
    init_scale = 0.1
    learning_rate = 0.1
    lr_decay = 0.9
    max_grad_norm = 5

    num_layers = 2
    embed_size = 200    # 50, 100, 200 or 300 to match glove embeddings
    hidden_size = 150

    max_sentence_length = 106
    max_shortsentence_length = 15

    vocab_size = 10000
    rel_vocab_size = 8

    dropout_keep_prob = 1.

    train_size = 0  # 0 to use all remaining data for training.
    validation_size = 1000
    test_size = 0

    training_steps = 300000
    batch_size = 100

    report_step = 100
    save_step = 2000

    srcfile = '/data/NYT/nyt-freebase.train.triples.universal.mention.txt'
    datafolder = '/data/train/'
    embedfolder = '/data/glove/'

    # model name - if provided, will seek to load previous checkpoint and continue training.
    modelname = 'testing'


def main():

    config = Config()

    if config.modelname:
        modelname = config.modelname
    else:
        modelname = time.strftime("%Y%m%d-%H%M%S")

    logfolder = '/tmp/tflogs/' + modelname + '/'
    ckptpath = '/tmp/tfckpt/' + modelname + '/'
    if not os.path.exists(ckptpath):
        os.makedirs(ckptpath)
    ckptfile = ckptpath + 'model.ckpt'

    # TODO: create ckptfolder if it doesn't exist!!

    # Create a saver for writing checkpoints.
    ckpt = tf.train.get_checkpoint_state(ckptpath)
    print ckpt

    print('LOAD DATA')
    datasets = get_datasets(config)
    print('%i training examples loaded' % datasets.train.num_examples)
    num_epochs = (1. * config.training_steps * config.batch_size) / datasets.train.num_examples
    print('%i steps of %i-batches = %f epochs' % (config.training_steps, config.batch_size, num_epochs))

    print('BUILD GRAPH')
    m = RNNClassifierModel(config=config)  # , init_embedding=init_embedding)

    # Op to generate summary stats
    merged = tf.merge_all_summaries()

    saver = tf.train.Saver()

    print('RUN TRAINING')

    with tf.Session() as sess:

        if ckpt:
            print('RESTORE VARIABLES FOR PREVIOUS MODEL')
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('INITIALISE VARIABLES FOR NEW MODEL')
            init_op = tf.initialize_all_variables()
            sess.run(init_op)
            print('APPLY INITIAL WORD EMBEDDINGS')
            init_embedding = build_initial_embedding(config)
            sess.run(m.embedding.assign(init_embedding))

        # instantiate SummaryWriters to output summaries and the Graph.
        writer = tf.train.SummaryWriter(logfolder + 'train/', graph=sess.graph)
        writer_val = tf.train.SummaryWriter(logfolder + 'val/')

        # keep track of validation costs to adjust learning rate when needed
        previous_val_costs = []

        # loop over training steps
        for step in range(config.training_steps):

            # get batch data and make feed_dict
            data_batch = datasets.train.next_batch(config.batch_size)
            feed_dict = make_feed_dict(m, data_batch, config.dropout_keep_prob)

            # run a training step
            sess.run(m.train_op, feed_dict=feed_dict)

            # write summaries
            summaries, cost_batch = sess.run([merged, m.cost], feed_dict=feed_dict)
            writer.add_summary(summaries, step)

            if step % config.report_step == 0:
                # get statistics on current batch
                print('step:%2i batch cost:%8f ' % (step, cost_batch))

                # get statistics on validation data - no dropout
                data_val = datasets.validation.data
                feed_dict = make_feed_dict(m, data_val, dropout_keep_prob=1.0)

                summaries_val, accuracy_val, cost_val = sess.run([merged, m.accuracy, m.cost],
                                                                 feed_dict=feed_dict)
                print('step:%2i val accuracy:%8f ' % (step, accuracy_val))
                print('step:%2i val cost:%8f ' % (step, cost_val))
                writer_val.add_summary(summaries_val, step)

                accuracy_byclass = sess.run(m.accuracy_byclass, feed_dict=feed_dict)
                print('class accuracy:')
                print(accuracy_byclass)
                pred_byclass = sess.run(m.pred_byclass, feed_dict=feed_dict)
                print('class prediction count:')
                print(pred_byclass)
                actual_byclass = sess.run(m.actual_byclass, feed_dict=feed_dict)
                print('class actual count:')
                print(actual_byclass)

                # decay learning rate if not improving
                if len(previous_val_costs) > 2 and cost_val > max(previous_val_costs[-3:]):
                    m.decay_lr(sess, config.lr_decay)
                    print('decayed lr to:', sess.run(m.lr, feed_dict))
                previous_val_costs.append(cost_val)

            if step % config.save_step == 0:
                print('Saving model to: %s' % ckptfile)
                saver.save(sess, ckptfile)

    print 'Optimisation complete'


def make_feed_dict(m, data, dropout_keep_prob=1.):
    feed_dict = {m.input_data: data['sentence'],
                 m.lengths: data['sentence_lengths'],
                 m.short_input_data: data['short'],
                 m.short_lengths: data['short_lengths'],
                 m.short_weights: data['short_weights'],
                 m.targets: data['relation'],
                 m.dropout_keep_prob: dropout_keep_prob}
    return feed_dict


if __name__ == "__main__":
    main()
