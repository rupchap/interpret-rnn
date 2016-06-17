import tensorflow as tf
import numpy as np
from load_data import read_data_sets
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn


# graph params
datafolder = '/home/rupchap/data/NYT/'
logfolder = 'tflogs/'

vocab_size = 10000
rel_vocab_size = 25

embed_size = 30
hidden_size = 5

validation_size = 5000
test_size = 500

learning_rate = 0.01
training_epochs = 20
batch_size = 500
display_step = 5
# TODO: bucket instead of padding all to max sentence length.
max_sentence_length = 104
PAD_ID = 1


def main():

    print('LOAD DATA')
    datasets = read_data_sets(datafolder, vocab_size, rel_vocab_size, validation_size, test_size)

    print('BUILD GRAPH')

    x = tf.placeholder(dtype='int32', shape=[None, max_sentence_length])
    y = tf.placeholder(dtype='int32', shape=[None, rel_vocab_size])

    inputs = embed(x)
    logits = inference(inputs)
    cost = loss(logits, y)

    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)

    train_op = training(cost, global_step)
    eval_op = evaluate(logits, y)

    # write graph to tensorboard
    # summary_op = tf.merge_all_summaries()
    # summary_writer = tf.train.SummaryWriter(logfolder, graph_def=sess.graph_def)

    print('RUN TRAINING')

    with tf.Session() as sess:
        # initialise
        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        for epoch in range(training_epochs):
            avg_cost = 0.

            num_batches = int(datasets.train.num_examples / batch_size)
            # Loop over batches
            for _ in range(num_batches):
                batch_x, batch_y = datasets.train.next_batch(batch_size)

                # pad sentences up to length of longest sentence in batch
                for sentence in batch_x:
                    sentence += [PAD_ID] * (max_sentence_length - len(sentence))
                batch_x = batch_x.tolist()

                # train on batch data
                feed_dict = {x: batch_x, y: batch_y}
                sess.run(train_op, feed_dict=feed_dict)

                # compute average loss
                batch_cost = sess.run(cost, feed_dict=feed_dict)
                avg_cost += batch_cost / num_batches


                # display logs
                # todo: make this do something!!
                # summary_str = sess.run(summary_op, feed_dict=feed_dict)
                # summary_writer.add_summary(summary_str, global_step=sess.run(global_step))

                # print 'global step:', global_step.eval()

            print('epoch:%2i avg_cost:%8f ' % (epoch, avg_cost))

    print 'Optimisation complete'


def embed(x):
    print 'embed - x:', x
    # Embedding
    with tf.device("/cpu:0"):
        embedding = tf.get_variable("embedding", [vocab_size, embed_size])
        inputs = tf.nn.embedding_lookup(embedding, x)
    print 'embed - inputs:', inputs
    return inputs


def inference(inputs, max_sentence_length=max_sentence_length,
              hidden_size=hidden_size, predict_classes=rel_vocab_size):
    lstm_cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)

    initial_state = lstm_cell.zero_state(batch_size, tf.float32)

    # TODO: should be possible to use the below rather than manually unrolling.
    # inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, max_sentence_length, inputs)]
    # outputs, state = rnn.rnn(lstm_cell, inputs, initial_state)
    # pick up last output - the only one we need.
    # output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])

    state = initial_state
    with tf.variable_scope("RNN"):
        for time_step in range(max_sentence_length):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            cell_output, state = lstm_cell(inputs[:, time_step, :], state)

    output = cell_output
    print 'inference - output:', output

    # TODO: Use linear function from TF??
    softmax_w = tf.get_variable("softmax_w", [hidden_size, predict_classes])
    softmax_b = tf.get_variable("softmax_b", [predict_classes])
    logits = tf.matmul(output, softmax_w) + softmax_b
    print 'inference - logits:', logits

    return logits


def loss(logits, y):
    y = tf.to_float(y)
    # note: could use tf.nn.sparse_softmax_cross_entropy_with_logits if y provided as index instead of onehot
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y, name='cross_entropy')
    loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

    print 'loss - logits:', logits
    print 'loss - y:', y
    print 'loss - loss', loss
    return loss


def training(cost, global_step):
    tf.scalar_summary('cost', cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op


def evaluate(output, y):
    correct_prediction = tf.equal(tf.arg_max(output, dimension=1),
                                  tf.arg_max(y, dimension=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


if __name__ == "__main__":
    main()
