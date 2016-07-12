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
        self._lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='lengths')
        self._short = tf.placeholder(dtype=tf.int32, shape=[None, config.max_shortsentence_length], name='shortdata')
        self._shortlengths = tf.placeholder(dtype=tf.int32, shape=[None], name='shortlengths')
        self._y = tf.placeholder(dtype=tf.int32, shape=[None], name='labels')

        # dropout placeholder
        self._dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # embedding
        with tf.device("/cpu:0"), tf.name_scope('Embedding'):
            if init_embedding is not None:
                embedding = tf.Variable(init_embedding, name='embedding')
            else:
                embedding = tf.get_variable("embedding", [config.vocab_size, config.embed_size])
            inputs = tf.nn.embedding_lookup(embedding, self._x)
            shortinputs = tf.nn.embedding_lookup(embedding, self._short)

        lengths = self._lengths

        # bi-RNN
        with tf.variable_scope("RNN"):
            # LSTM cell
            cell_fw = rnn_cell.BasicLSTMCell(config.hidden_size, forget_bias=1.0, state_is_tuple=True)
            cell_bw = rnn_cell.BasicLSTMCell(config.hidden_size, forget_bias=1.0, state_is_tuple=True)
            # Dropout
            cell_fw = rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=self._dropout_keep_prob)
            cell_bw = rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=self._dropout_keep_prob)
            # Multilayer
            cell_fw = rnn_cell.MultiRNNCell([cell_fw] * config.num_layers, state_is_tuple=True)
            cell_bw = rnn_cell.MultiRNNCell([cell_bw] * config.num_layers, state_is_tuple=True)
            # RNN
            inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, config.max_sentence_length, inputs)]

            outputs, output_state_fw, output_state_bw = tf.nn.bidirectional_rnn(cell_fw, cell_bw, inputs,
                                                                                dtype=tf.float32,
                                                                                sequence_length=lengths)

            # TODO - fix so not rely on hardcoded number of  RNN layers
            state_0 = tf.concat(1, [output_state_fw[0].c, output_state_bw[0].c])
            output_1 = tf.concat(1, [output_state_fw[1].h, output_state_bw[1].h])
            output = output_1

        # Linear transform - full RNN to relation
        with tf.name_scope('Logits'):
            W = tf.get_variable("W", [config.hidden_size*2, config.rel_vocab_size])
            b = tf.get_variable("b", [config.rel_vocab_size])
            logits = tf.matmul(output, W) + b

        # Cost
        # TODO: add in cost element arising from prediction of short sentence.
        # TODO: need to add _EOS tag to end of short sentences?
        with tf.name_scope('Cost'):
            # cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits, self._y, name='cross_entropy')
            cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self._y, name='cross_entropy')

            cost = tf.reduce_mean(cross_entropy_loss, name='cross_entropy_mean')

            self._cost = cost

            tf.scalar_summary('cost', cost)

            #  TODO: just feeding FW state from bidirectional RNN for now - should also use BW state?
        with tf.variable_scope("ShortSequenceDecoder"):
            cell_dc = rnn_cell.BasicLSTMCell(config.hidden_size, forget_bias=1.0, state_is_tuple=True)
            inputs_dc = tf.zeros_like(shortinputs)
            inputs_dc = [tf.squeeze(input_, [1]) for input_ in tf.split(1, config.max_shortsentence_length, inputs_dc)]
            outputs_dc, state_dc = tf.nn.seq2seq.rnn_decoder(decoder_inputs=inputs_dc,
                                                             initial_state=output_state_fw[0],
                                                             cell=cell_dc)

        with tf.variable_scope("ShortSequenceLogits"):
        # TODO: calc. logits for predicted shortsentence sequence

        with tf.variable_scope('ShortSequenceCost'):
        # TODO: calc. cost or predicted sequence vs. actual short sentence

        with tf.variable_scope('TotalCost'):
        # TODO: calc. total cost for subsequent use in training
        # TODO: how to incorporate dropouts where only either sh. sentence or relation is provided for training?

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
    def lengths(self):
        return self._lengths

    @property
    def short_input_data(self):
        return self._short

    @property
    def short_lengths(self):
        return self._shortlengths

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

