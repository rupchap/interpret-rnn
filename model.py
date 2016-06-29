from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn


class RNNClassifierModel(object):

    def __init__(self, is_training, config):

        # placeholders
        self._x = tf.placeholder(dtype=tf.int32, shape=[None, config.max_sentence_length], name='data')
        # self._y = tf.placeholder(dtype=tf.int32, shape=[None, config.rel_vocab_size], name='labels')
        self._y = tf.placeholder(dtype=tf.int32, shape=[None], name='labels')
        

        # embedding
        with tf.name_scope('Embedding'):
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [config.vocab_size, config.embed_size])
                inputs = tf.nn.embedding_lookup(embedding, self._x)

        # embedding dropout
        if is_training and config.keep_prob < 1:
            with tf.name_scope('EmbeddingDropout'):
                inputs = tf.nn.dropout(inputs, config.keep_prob)

        # RNN
        with tf.variable_scope("RNN"):

            # LSTM cell
            cell = rnn_cell.BasicLSTMCell(config.hidden_size, forget_bias=1.0)

            # LSTM dropout
            if is_training and config.keep_prob < 1:
                cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=config.keep_prob)

            # Multilayer
            cell = rnn_cell.MultiRNNCell([cell] * config.num_layers)

            # RNN
            inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, config.max_sentence_length, inputs)]
            outputs, state = rnn.rnn(cell, inputs, dtype=tf.float32)
            output = outputs[-1]
            self._final_state = state


        # Linear transform
        with tf.name_scope('LinearTransform'):
            W = tf.get_variable("W", [config.hidden_size, config.rel_vocab_size])
            b = tf.get_variable("b", [config.rel_vocab_size])
            logits = tf.matmul(output, W) + b

        # Cost
        with tf.name_scope('CrossEntropy'):
            # y = tf.to_float(self._y)
            # cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits, y, name='cross_entropy')
            y = tf.to_int64(self._y)
            cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y, name='cross_entropy')
            cost = tf.reduce_mean(cross_entropy_loss, name='cross_entropy_mean')
            tf.scalar_summary('cost', cost)
            self._cost = cost


        # Predict and assess accuracy
        with tf.name_scope('Accuracy'):
            y_proba = tf.nn.softmax(logits)
            y_pred = tf.arg_max(logits, dimension=1)
            # correct_prediction = tf.equal(y_pred, tf.arg_max(y, dimension=1))
            correct_prediction = tf.equal(y_pred, y)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary('accuracy', accuracy)

        self._proba = y_proba
        self._ypred = y_pred
        self._accuracy = accuracy

        # add optimizer and learning rate for training model only.
        if is_training:

            # Initial learning rate
            self._lr = tf.Variable(config.learning_rate, trainable=False)

            # Optimizer with clipped gradients
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self._train_op = optimizer.apply_gradients(zip(grads, tvars))

        # Merge summary ops
        # merge_op = tf.merge_all_summaries()
        # self._merge_op = merge_op

        return

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def decay_lr(self, session, lr_decay):
        new_lr = self.lr * lr_decay
        self.assign_lr(session, new_lr)

    @property
    def input_data(self):
        return self._x

    @property
    def targets(self):
        return self._y

    @property
    def predictions(self):
        return self._ypred

    @property
    def proba(self):
        return self._proba

    @property
    def cost(self):
        return self._cost

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def merge_op(self):
        return self._merge_op
