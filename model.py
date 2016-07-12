from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn


class RNNClassifierModel(object):

    def __init__(self, config, init_embedding=None):

        # data placeholders
        self._x = tf.placeholder(dtype=tf.int32, shape=[None, config.max_sentence_length], name='data')
        # self._y = tf.placeholder(dtype=tf.float32, shape=[None, config.rel_vocab_size], name='labels')
        self._y = tf.placeholder(dtype=tf.int32, shape=[None], name='labels')
        self._lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='lengths')

        # dropout placeholder
        self._dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # embedding
        with tf.device("/cpu:0"), tf.name_scope('Embedding'):
            if init_embedding is not None:
                embedding = tf.Variable(init_embedding, name='embedding')
            else:
                embedding = tf.get_variable("embedding", [config.vocab_size, config.embed_size])
            inputs = tf.nn.embedding_lookup(embedding, self._x)

        lengths = self._lengths

        # bi-RNN
        with tf.variable_scope("RNN"):
            # LSTM cell
            cell_fw = rnn_cell.BasicLSTMCell(config.hidden_size, forget_bias=1.0)
            cell_bw = rnn_cell.BasicLSTMCell(config.hidden_size, forget_bias=1.0)
            # Dropout
            cell_fw = rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=self._dropout_keep_prob)
            cell_bw = rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=self._dropout_keep_prob)
            # Multilayer
            cell_fw = rnn_cell.MultiRNNCell([cell_fw] * config.num_layers)
            cell_bw = rnn_cell.MultiRNNCell([cell_bw] * config.num_layers)
            # RNN
            inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, config.max_sentence_length, inputs)]

            outputs, output_state_fw, output_state_bw = tf.nn.bidirectional_rnn(cell_fw, cell_bw, inputs,
                                                                                dtype=tf.float32,
                                                                                sequence_length=lengths)

            output = tf.concat(1, [output_state_fw, output_state_bw])

        # Linear transform
        with tf.name_scope('Logits'):
            W = tf.get_variable("W", [config.hidden_size*4, config.rel_vocab_size])
            b = tf.get_variable("b", [config.rel_vocab_size])
            logits = tf.matmul(output, W) + b

        # Cost
        with tf.name_scope('Cost'):
            # cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits, self._y, name='cross_entropy')
            cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self._y, name='cross_entropy')

            cost = tf.reduce_mean(cross_entropy_loss, name='cross_entropy_mean')

            self._cost = cost

            tf.scalar_summary('cost', cost)

        # Predict and assess accuracy
        with tf.name_scope('Accuracy'):
            y_proba = tf.nn.softmax(logits)
            y_pred = tf.arg_max(logits, dimension=1)
            y_pred = tf.cast(y_pred, tf.int32)
            # y_actual = tf.arg_max(self._y, dimension=1)
            y_actual = self._y
            correct_prediction = tf.equal(y_pred, y_actual)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            self._proba = y_proba
            self._ypred = y_pred
            self._accuracy = accuracy

            # accuracy per class
            y_pred_onehot = tf.one_hot(y_pred, depth=config.rel_vocab_size, dtype=tf.float32)
            # y_actual_onehot = self._y
            y_actual_onehot = tf.one_hot(self._y, depth=config.rel_vocab_size, dtype=tf.float32)
            y_correct_onehot = tf.mul(y_actual_onehot, y_pred_onehot)
            accuracy_byclass = tf.reduce_mean(y_correct_onehot, 0)
            pred_byclass = tf.reduce_sum(y_pred_onehot, 0)
            actual_byclass = tf.reduce_sum(y_actual_onehot, 0)
            self._accuracy_byclass = accuracy_byclass
            self._pred_byclass = pred_byclass
            self._actual_byclass = actual_byclass

            tf.scalar_summary('accuracy', accuracy)

        # Initial learning rate
        self._lr = tf.Variable(config.learning_rate, trainable=False)

        # Optimizer with clipped gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

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
    def lengths(self):
        return self._lengths

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
        return self._accuracy\

    @property
    def accuracy_byclass(self):
        return self._accuracy_byclass

    @property
    def pred_byclass(self):
        return self._pred_byclass

    @property
    def actual_byclass(self):
        return self._actual_byclass

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def dropout_keep_prob(self):
        return self._dropout_keep_prob

    @property
    def train_op(self):
        return self._train_op

