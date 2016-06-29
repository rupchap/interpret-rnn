import tensorflow as tf
import numpy as np
from load_data import read_data_sets
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn
import time

from model import RNNClassifierModel

# To run tensorboard:
# tensorboard --logdir=/tmp/tflogs

# file locations
datafolder = '/data/NYT/'
logfolder = '/tmp/tflogs/' + '123/'
save_path = '/tmp/tfmodel.ckpt'


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 0.1
    lr_decay = 0.9
    max_grad_norm = 5
    num_layers = 1
    keep_prob = 1.0
    vocab_size = 7000
    hidden_size = 150
    max_sentence_length = 104
    rel_vocab_size = 12
    embed_size = 100

    dropout_keep_prob = 0.9

    validation_size = 5000
    test_size = 500

    training_steps = 300000
    batch_size = 100
    checkpoint_step = 10
    save_step = 1000


def main():

    config = SmallConfig()

    print('LOAD DATA')
    datasets = read_data_sets(datafolder, config.vocab_size, config.rel_vocab_size,
                              config.validation_size, config.test_size)
    print('%i training examples loaded' % datasets.train.num_examples)
    num_epochs = (1.0 * config.training_steps * config.batch_size) / datasets.train.num_examples
    print('%i steps of %i-batches = %f epochs' % (config.training_steps, config.batch_size, num_epochs))

    print('BUILD GRAPH')
    m = RNNClassifierModel(config=config)

    # Op to generate summary stats
    merged = tf.merge_all_summaries()
    print merged

    # Create a saver for writing checkpoints.
    saver = tf.train.Saver()

    print('RUN TRAINING')

    with tf.Session() as sess:
        # initialise variables
        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        # instantiate SummaryWriters to output summaries and the Graph.
        writer = tf.train.SummaryWriter(logfolder + 'train/', graph_def=sess.graph_def)
        writer_val = tf.train.SummaryWriter(logfolder + 'val/')

        # keep track of validation costs to adjust learning rate when needed
        previous_val_costs = []

        # loop over training steps
        for step in range(config.training_steps):

            # get batch data
            x_batch, y_batch = datasets.train.next_batch(config.batch_size)
            x_batch = x_batch.tolist()
            feed_dict = {m.input_data: x_batch,
                         m.targets: y_batch,
                         m.dropout_keep_prob: config.dropout_keep_prob}

            # run a training step
            sess.run(m.train_op, feed_dict=feed_dict)

            if step % config.checkpoint_step == 0:
                # get statistics on current batch
                summaries, cost_batch = sess.run([merged, m.cost], feed_dict=feed_dict)
                print('step:%2i batch cost:%8f ' % (step, cost_batch))
                writer.add_summary(summaries, step)

                # TODO: consider splitting accuracy by relation type.
                # TODO: hook up to Sebastien's prediction accuracy - may need to flip to a generative model??

                # get statistics on validation data
                x_val, y_val = datasets.validation.get_all()
                x_val = x_val.tolist()
                feed_dict = {m.input_data: x_val,
                             m.targets: y_val,
                             m.dropout_keep_prob: 0.9}

                summaries_val, accuracy_val, cost_val = sess.run([merged, m.accuracy, m.cost],
                                                                 feed_dict=feed_dict)
                print('step:%2i val accuracy:%8f ' % (step, accuracy_val))
                print('step:%2i val cost:%8f ' % (step, cost_val))
                writer_val.add_summary(summaries_val, step)

                # decay learning rate if not improving
                if len(previous_val_costs) > 2 and cost_val > max(previous_val_costs[-3:]):
                    m.decay_lr(sess, 0.9)
                    print('decayed lr to:', sess.run(m.lr, feed_dict))
                previous_val_costs.append(cost_val)

                saver.save(sess, save_path=save_path)
                print('Model saved in: %s' % save_path)

    print 'Optimisation complete'


if __name__ == "__main__":
    main()
