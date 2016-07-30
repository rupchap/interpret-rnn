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
        self._shortweights = tf.placeholder(dtype=tf.float32, shape=[None, config.max_shortsentence_length], name='shortweights')
        self._y = tf.placeholder(dtype=tf.int32, shape=[None], name='labels')

        # dropout placeholder
        self._dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # build input for short sentence decoder - just _go symbol [2] followed by _pad [0]
        batch_size = tf.shape(self._x)[0]
        pads = tf.zeros([batch_size, config.max_shortsentence_length - 1], dtype=tf.int32)
        gos = tf.fill([batch_size, 1], 2)
        shortinputs = tf.concat(1, [gos, pads])

        # embedding
        with tf.device("/cpu:0"), tf.name_scope('Embedding'):
            if init_embedding is not None:
                embedding = tf.Variable(init_embedding, name='embedding')
            else:
                embedding = tf.get_variable("embedding", [config.vocab_size, config.embed_size])
            inputs = tf.nn.embedding_lookup(embedding, self._x)

            shortgolds = tf.nn.embedding_lookup(embedding, self._short)
            shortinputs_embed = tf.nn.embedding_lookup(embedding, shortinputs)

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

            # convert inputs tensor to list of 2d tensors [batch_size * embed_size] of length max_sentence_length
            inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, config.max_sentence_length, inputs)]

            # RNN
            outputs, output_state_fw, output_state_bw = tf.nn.bidirectional_rnn(cell_fw, cell_bw, inputs,
                                                                                dtype=tf.float32,
                                                                                sequence_length=lengths)

            # concatenate fw and bw output states
            output_state_comb = [tf.concat(1, [output_state_fw[i], output_state_bw[i]])
                                 for i in range(len(output_state_fw))]

            # TODO - fix so not rely on hardcoded number of  RNN layers
            state_0 = tf.concat(1, [output_state_fw[0].c, output_state_bw[0].c])
            output_1 = tf.concat(1, [output_state_fw[1].h, output_state_bw[1].h])
            output = output_1

        # DECODE RELATION
        # Linear transform - final RNN state to relation
        with tf.name_scope('Logits'):
            W_rel = tf.get_variable("W_rel", [config.hidden_size*2, config.rel_vocab_size])
            b_rel = tf.get_variable("b_rel", [config.rel_vocab_size])
            logits_rel = tf.matmul(output_1, W_rel) + b_rel

        # RelationCost
        with tf.name_scope('RelationCost'):
            cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_rel, self._y, name='cross_entropy')
            cost_relation = tf.reduce_mean(cross_entropy_loss, name='cross_entropy_mean')
            self._cost_relation = cost_relation
            tf.scalar_summary('cost_relation', cost_relation)

        # DECODE SHORT SENTENCE
        # TODO: just feeding FW state from bidirectional RNN for now - should also use BW state?
        with tf.variable_scope("ShortSequenceDecoder"):
            cell_dc = rnn_cell.BasicLSTMCell(config.hidden_size, forget_bias=1.0, state_is_tuple=True)
            shortinputs_embed = [tf.squeeze(input_, [1]) for input_ in
                                 tf.split(1, config.max_shortsentence_length, shortinputs_embed)]

            outputs_dc, state_dc = tf.nn.seq2seq.rnn_decoder(decoder_inputs=shortinputs_embed,
                                                             initial_state=output_state_fw[0],
                                                             cell=cell_dc)

        # TODO: can we use the embedding matrix here somehow?
        with tf.variable_scope("ShortSequenceLogits"):
            W_short = tf.get_variable("W_short", [config.hidden_size, config.vocab_size])
            b_short = tf.get_variable("b_short", [config.vocab_size])
            logits_short = [tf.matmul(output, W_short) + b_short for output in outputs_dc]

        with tf.variable_scope('ShortSequenceCost'):
            targets = [tf.squeeze(input_, [1]) for input_ in tf.split(1, config.max_shortsentence_length, self._short)]
            weights = [tf.squeeze(weight_, [1]) for weight_ in tf.split(1, config.max_shortsentence_length, self._shortweights)]

            cost_short = tf.nn.seq2seq.sequence_loss(logits=logits_short,
                                                     targets=targets,
                                                     weights=weights)
            self._cost_short = cost_short
            tf.scalar_summary('cost_short', cost_short)

        with tf.variable_scope('TotalCost'):
            cost = cost_relation + cost_short
            self._cost = cost

        # Predict and assess accuracy
        with tf.name_scope('Accuracy'):
            y_proba = tf.nn.softmax(logits_rel)
            y_pred = tf.arg_max(logits_rel, dimension=1)
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
    def short_weights(self):
        return self._shortweights

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

