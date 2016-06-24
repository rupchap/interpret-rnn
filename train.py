import tensorflow as tf
import numpy as np
from load_data import read_data_sets
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn
import time


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
training_steps = 100000
batch_size = 50
display_step = 1
save_step = 1000

max_sentence_length = 104


def main():

    print('LOAD DATA')
    datasets = read_data_sets(datafolder, vocab_size, rel_vocab_size, validation_size, test_size)
    print('%i training examples loaded' % datasets.train.num_examples)
    num_epochs = (1.0 * training_steps * batch_size) / datasets.train.num_examples
    print('%i steps of %i-batches = %f epochs' % (training_steps, batch_size, num_epochs))

    print('BUILD GRAPH')
    x = tf.placeholder(dtype='int32', shape=[None, max_sentence_length], name='data')
    y = tf.placeholder(dtype='int32', shape=[None, rel_vocab_size], name='labels')

    # todo: either make this a contant or add to all dicts.
    dropout = tf.placeholder(tf.float32)

    inputs = embed(x)
    logits = inference(inputs, dropout)
    cost = loss(logits, y)
    train_op = training(cost)

    accuracy = evaluate(logits, y)

    # Op to generate summary stats
    summary_op = tf.merge_all_summaries()

    # Create a saver for writing checkpoints.
    saver = tf.train.Saver()

    print('RUN TRAINING')

    with tf.Session() as sess:
        # initialise
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        # Instantiate a SummaryWriter to output summaries and the Graph.
        train_writer = tf.train.SummaryWriter(logfolder+ '/train', graph_def=sess.graph_def)
        validation_writer = tf.train.SummaryWriter(logfolder + '/validation')


        dropout = 0.8

        for step in range(training_steps):

            batch_x, batch_y = datasets.train.next_batch(batch_size)
            batch_x = batch_x.tolist()

            # train on batch data
            feed_dict = {x: batch_x, y: batch_y}
            sess.run(train_op, feed_dict=feed_dict)

            if step % display_step == 0:
                # print current batch cost
                batch_cost = sess.run(cost, feed_dict=feed_dict)
                print('step:%2i batch cost:%8f ' % (step, batch_cost))

                # print validation accuracy

                # TODO: consider splitting accuracy by relation type.
                # TODO: hook up to Sebastien's prediction accuracy - may need to flip to a generative model??

                validation_x, validation_y = datasets.validation.get_all()
                validation_x = validation_x.tolist()
                validation_dict = {x: validation_x,
                                   y: validation_y}
                val_batch_accuracy = sess.run(accuracy, feed_dict=validation_dict)
                print('step:%2i val accuracy:%8f ' % (step, val_batch_accuracy))

                # Update the events file for validation
                summary_str = sess.run(summary_op, feed_dict=validation_dict)
                validation_writer.add_summary(summary_str, step)


            if step % save_step == 0:
                # save variables
                saver.save(sess, save_path=save_path)
                print('Model saved in: %s' % save_path)

            # Update the events file for training
            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            train_writer.add_summary(summary_str, step)


    print 'Optimisation complete'


def embed(x):
    print 'embed - x:', x
    # Embedding
    with tf.name_scope('Embedding'):
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, embed_size])
            inputs = tf.nn.embedding_lookup(embedding, x)
    print 'embed - inputs:', inputs
    return inputs


def inference(inputs, dropout, max_sentence_length=max_sentence_length,
              hidden_size=hidden_size, predict_classes=rel_vocab_size):

    with tf.variable_scope("RNN"):

        # create basic LSTM cell
        cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)

        # add dropout wrapper
        cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)

        # add multiple layers
        cell = rnn_cell.MultiRNNCell([cell] * num_layers)

        # TODO: consider using tf.unpack() here for readability?
        inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, max_sentence_length, inputs)]
        print 'inference - inputs:', inputs
        outputs, state = rnn.rnn(cell, inputs, dtype=tf.float32)
        # pick up last output - the only one we need.
        output = outputs[-1]

    print 'inference - output:', output

    with tf.name_scope('LinearTransform'):
        W = tf.get_variable("W", [hidden_size, predict_classes])
        b = tf.get_variable("b", [predict_classes])
        logits = tf.matmul(output, W) + b
    print 'inference - logits:', logits

    return logits


def loss(logits, y):
    # note: could use tf.nn.sparse_softmax_cross_entropy_with_logits if y provided as index instead of onehot
    with tf.name_scope('CrossEntropy'):
        y = tf.to_float(y)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y, name='cross_entropy')
        loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

    print 'loss - logits:', logits
    print 'loss - y:', y
    print 'loss - loss', loss
    return loss


def training(cost):
    tf.scalar_summary('cost', cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cost)
    return train_op


def evaluate(output, y):
    with tf.name_scope('Evaluate'):
        correct_prediction = tf.equal(tf.arg_max(output, dimension=1),
                                      tf.arg_max(y, dimension=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy', accuracy)
    return accuracy


if __name__ == "__main__":
    main()
