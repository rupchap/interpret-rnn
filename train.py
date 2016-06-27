import tensorflow as tf
import numpy as np
from load_data import read_data_sets
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn
import time

from model import RNNClassifierModel
from model import SmallConfig

# To run tensorboard:
# tensorboard --logdir=/tmp/tflogs

# file locations
datafolder = '/data/NYT/'
logfolder = '/tmp/tflogs/' + time.strftime("%Y%m%d-%H%M%S") + '/'
save_path = '/tmp/tfmodel.ckpt'

# graph params
vocab_size = 10000
rel_vocab_size = 25

embed_size = 30
hidden_size = 50
num_layers = 3

validation_size = 5000
test_size = 500

learning_rate = 0.01
dropout_keep_prob = 0.8
training_steps = 100000
batch_size = 50
checkpoint_step = 100
save_step = 1000

max_sentence_length = 104


def main():

    print('LOAD DATA')
    datasets = read_data_sets(datafolder, vocab_size, rel_vocab_size, validation_size, test_size)
    print('%i training examples loaded' % datasets.train.num_examples)
    num_epochs = (1.0 * training_steps * batch_size) / datasets.train.num_examples
    print('%i steps of %i-batches = %f epochs' % (training_steps, batch_size, num_epochs))

    print('BUILD GRAPH')
    # Build training model
    train_config = SmallConfig()
    with tf.variable_scope("model"):
        m = RNNClassifierModel(is_training=True, config=train_config)

    # Build validation model, reusing training variables
    val_config = train_config
    with tf.variable_scope("model", reuse=True):
        m_val = RNNClassifierModel(is_training=False, config=val_config)

    # Op to generate summary stats
    merged = tf.merge_all_summaries()

    # Create a saver for writing checkpoints.
    # saver = tf.train.Saver()

    print('RUN TRAINING')

    with tf.Session() as sess:
        # initialise
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        # Instantiate a SummaryWriter to output summaries and the Graph.
        writer = tf.train.SummaryWriter(logfolder + '/train', graph_def=sess.graph_def)


        previous_val_costs = []
        for step in range(training_steps):

            # train on batch data
            x_batch, y_batch = datasets.train.next_batch(batch_size)
            x_batch = x_batch.tolist()
            feed_dict = {m.input_data: x_batch,
                         m.targets: y_batch}

            sess.run(m.train_op, feed_dict=feed_dict)

            if step % checkpoint_step == 0:
                # print current batch cost
                batch_cost = sess.run(m.cost, feed_dict=feed_dict)
                print('step:%2i batch cost:%8f ' % (step, batch_cost))


                # TODO: consider splitting accuracy by relation type.
                # TODO: hook up to Sebastien's prediction accuracy - may need to flip to a generative model??

                # print validation accuracy
                x_val, y_val = datasets.validation.get_all()
                x_val = x_val.tolist()
                feed_dict_val = {m_val.input_data: x_val,
                                 m_val.targets: y_val}
                accuracy_val = sess.run(m_val.accuracy, feed_dict=feed_dict_val)
                cost_val = sess.run(m_val.cost, feed_dict=feed_dict_val)
                print('step:%2i val accuracy:%8f ' % (step, accuracy_val))
                print('step:%2i val cost:%8f ' % (step, cost_val))
                # print 'targets:', y_val
                # print 'proba:', sess.run(m_val.proba, feed_dict_val)
                # print 'predict:', sess.run(m_val.predictions, feed_dict_val)

                # decay learning rate if not improving
                if len(previous_val_costs) > 2 and cost_val > max(previous_val_costs[-3:]):
                    m.decay_lr(sess, 0.9)
                    print 'decayed lr to:', sess.run(m.lr, feed_dict)
                previous_val_costs.append(cost_val)

                # Update the events file for validation
                # summary_str = sess.run(summary_op, feed_dict=feed_dict_val)
                # writer_val.add_summary(summary_str, step)
                # summary_str = sess.run(summary_op, feed_dict=feed_dict)
                # writer.add_summary(summary_str, step)


            # if step % save_step == 0:
            #     # save variables
            #     saver.save(sess, save_path=save_path)
            #     print('Model saved in: %s' % save_path)
            #
            # # Update the events file for training
            # summary_str = sess.run(summary_op, feed_dict=feed_dict)
            # train_writer.add_summary(summary_str, step)

    print 'Optimisation complete'


if __name__ == "__main__":
    main()
