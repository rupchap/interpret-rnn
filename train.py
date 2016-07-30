import tensorflow as tf
import numpy as np
from load_data import get_datasets
from load_data import build_initial_embedding
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
import time

from model import RNNClassifierModel
from datasets import *

# To run tensorboard:
# tensorboard --logdir=/tmp/tflogs


class Config(object):
    init_scale = 0.1
    learning_rate = 0.1
    lr_decay = 1
    max_grad_norm = 50
    num_layers = 2
    keep_prob = 1.0
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
    report_step = 500
    save_step = 10000

    srcfile = '/data/NYT/nyt-freebase.train.triples.universal.mention.txt'
    datafolder = '/data/train/'
    logfolder = '/tmp/tflogs/' + time.strftime("%Y%m%d-%H%M%S") + '/'
    save_path = '/tmp/tfmodel.ckpt'
    embedfolder = '/data/glove/'


def main():

    config = Config()

    print('LOAD DATA')
    datasets = get_datasets(config)
    print('%i training examples loaded' % datasets.train.num_examples)
    num_epochs = (1.0 * config.training_steps * config.batch_size) / datasets.train.num_examples
    print('%i steps of %i-batches = %f epochs' % (config.training_steps, config.batch_size, num_epochs))

    print('GET INITIAL WORD EMBEDDINGS')
    init_embedding = build_initial_embedding(config)

    print('BUILD GRAPH')
    m = RNNClassifierModel(config=config, init_embedding=init_embedding)

    # Op to generate summary stats
    merged = tf.merge_all_summaries()

    # Create a saver for writing checkpoints.
    saver = tf.train.Saver()

    print('RUN TRAINING')

    with tf.Session() as sess:
        # initialise variables
        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        # instantiate SummaryWriters to output summaries and the Graph.
        logfolder = config.logfolder
        writer = tf.train.SummaryWriter(logfolder + 'train/', graph=sess.graph)
        writer_val = tf.train.SummaryWriter(logfolder + 'val/')

        # keep track of validation costs to adjust learning rate when needed
        previous_val_costs = []

        # loop over training steps
        for step in range(config.training_steps):

            # get batch data
            x_batch, lengths_batch, short_batch, short_lengths_batch, short_weights_batch, y_batch =\
                datasets.train.next_batch(config.batch_size)
            feed_dict = {m.input_data: x_batch,
                         m.lengths: lengths_batch,
                         m.short_input_data: short_batch,
                         m.short_lengths: short_lengths_batch,
                         m.short_weights: short_weights_batch,
                         m.targets: y_batch,
                         m.dropout_keep_prob: config.dropout_keep_prob}

            # run a training step
            sess.run(m.train_op, feed_dict=feed_dict)

            # write summaries
            summaries, cost_batch = sess.run([merged, m.cost], feed_dict=feed_dict)
            writer.add_summary(summaries, step)

            if step % config.report_step == 0:
                # get statistics on current batch
                print('step:%2i batch cost:%8f ' % (step, cost_batch))

                # TODO: hook up to Sebastien's prediction accuracy - may need to flip to a generative model??

                # get statistics on validation data - no dropout
                x_val, lengths_val, short_val, short_lengths_val, short_weights_val, y_val =\
                    datasets.validation.get_all()
                feed_dict = {m.input_data: x_val,
                             m.lengths: lengths_val,
                             m.short_input_data: short_val,
                             m.short_lengths: short_lengths_val,
                             m.short_weights: short_weights_val,
                             m.targets: y_val,
                             m.dropout_keep_prob: 1.}

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
                    m.decay_lr(sess, 0.9)
                    print('decayed lr to:', sess.run(m.lr, feed_dict))
                previous_val_costs.append(cost_val)

            # if step % config.save_step == 0:
            #     save_path = config.save_step
            #     saver.save(sess, save_path=save_path)
            #     print('Model saved in: %s' % save_path)

    print 'Optimisation complete'


if __name__ == "__main__":
    # m = RNNClassifierModel(config=Config())
    #
    main()
