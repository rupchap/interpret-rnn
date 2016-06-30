
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import ast
import numpy as np
import pandas as pd
import cPickle as pickle
from collections import Counter

sourcefilename = 'nyt-freebase.train.triples.universal.mention.txt'

# regex for digits
_DIGIT_RE = re.compile(r"\d")
_WORD_SPLIT = ' '

# Special vocabulary symbols
_UNK = '_UNK'
_PAD = '_PAD'
_GO = '_GO'
_EOS = '_EOS'
_START_VOCAB = [_UNK, _PAD, _GO, _EOS]

UNK_ID = 0
PAD_ID = 1
GO_ID = 2
EOS_ID = 3

# Symbols for masked entities
_ENTA = '_ENTA'
_ENTB = '_ENTB'

# We will pad all sentences with _PAD up to max_sentence_length
# RNN in tensorflow currently requires constant sentence length
max_sentence_length = 104


def main():
    datasets = read_data_sets()
    sen, rel = datasets.train.next_batch(5)
    print('training sample:')
    for s in sen:
        print(s)
    for r in rel:
        print(r)

    sen, rel = datasets.test.next_batch(5)
    print('training sample:')
    for s in sen:
        print(s)
    for r in rel:
        print(r)


class DataSet(object):

    def __init__(self, sentences, relations):
        """Construct a DataSet.
        """

        # Check matching number of examples
        assert sentences.shape[0] == relations.shape[0], (
            'sentences.shape: %s relations.shape: %s' % (len(sentences), relations.shape))

        self._num_examples = sentences.shape[0]
        self._sentences = sentences
        self._relations = relations
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def sentences(self):
        return self._sentences

    @property
    def relations(self):
        return self._relations

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._sentences = self._sentences[perm]
            self._relations = self._relations[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch

        # return batch
        return self._sentences[start:end], self._relations[start:end]

    def get_all(self):
        """Return all examples from this data set."""

        return self._sentences, self._relations


def read_data_sets(datafolder='/data/NYT/',
                   vocab_size=10000, rel_vocab_size=25,
                   validation_size=5000, test_size=500,
                   train_size=0):
    class DataSets(object):
        pass
    data_sets = DataSets()

    # check if pickled data already exist
    picklefilepath = datafolder + 'data_%i_%i.pkl' % (vocab_size, rel_vocab_size)
    if os.path.isfile(picklefilepath):
        print('load pre-processed data')
        with open(picklefilepath, 'r') as f:
            data = pickle.load(f)
        relations = data['relations']
        sentences = data['sentences']

    else:
        print('no pre-processed datafiles found: rebuild them')
        relations, sentences = process_data(datafolder, vocab_size, rel_vocab_size)

        print('pickle for reuse later')
        data = {'relations': relations, 'sentences': sentences}
        with open(picklefilepath, 'w') as f:
            pickle.dump(data, f)

    # slice for training, validation, test datasets
    num_examples = sentences.shape[0]
    if not train_size:
        train_size = num_examples - validation_size - test_size

    sentences_val, sentences = np.split(sentences, [validation_size])
    sentences_test, sentences = np.split(sentences, [test_size])
    sentences_train, sentences = np.split(sentences, [train_size])

    relations_val, relations = np.split(relations, [validation_size])
    relations_test, relations = np.split(relations, [test_size])
    relations_train, relations = np.split(relations, [train_size])

    data_sets.train = DataSet(sentences_train, relations_train)
    data_sets.validation = DataSet(sentences_val, relations_val)
    data_sets.test = DataSet(sentences_test, relations_test)

    return data_sets


def filter_data(data):

    # unpack data tuple
    labels, relations, sentences, entAs, entBs = data

    pos_count = labels.count('POSITIVE')
    neg_count = labels.count('NEGATIVE')
    unl_count = labels.count('UNLABELED')
    print('data includes %i POSITIVE, %i NEGATIVE, %i UNLABELED.'
          % (pos_count, neg_count, unl_count))

    # Filter dataset to equal numbers of positive and negative cases. drop UNLABELED
    data = pd.DataFrame({'labels': labels, 'relations': relations,
                         'sentences': sentences, 'entAs': entAs, 'entBs': entBs})
    pos = data[data['labels'] == 'POSITIVE']
    neg = data[data['labels'] == 'NEGATIVE'].sample(n=pos_count)
    data_filtered = pd.concat([pos, neg])
    # shuffle rows
    data_filtered = data_filtered.sample(frac=1)

    print('filtered dataset created')

    labels = data_filtered['labels']
    relations = data_filtered['relations']
    sentences = data_filtered['sentences']
    entAs = data_filtered['entAs']
    entBs = data_filtered['entBs']

    return labels, relations, sentences, entAs, entBs


def process_data(datafolder='/data/NYT/', vocab_size=10000, rel_vocab_size=25):

    print('IMPORT DATA')
    # labels, relations, sentences, entAs, entBs = read_data(datafolder+sourcefilename)
    data = read_data(datafolder+sourcefilename)

    print('imported %i examples from %s' % (len(data[0]), sourcefilename))

    print('FILTER DATA')
    labels, relations, sentences, entAs, entBs = filter_data(data)

    print('PROCESS SENTENCES')
    print('parse sentences to mask entities')
    masked_sentences = [mask_entities_in_sentence(sentence, entA, entB)
                        for sentence, entA, entB in zip(sentences, entAs, entBs)]

    print('get vocab')
    vocabfilename = 'vocab_%i.txt' % vocab_size
    vocabfilepath = datafolder + vocabfilename
    if os.path.isfile(vocabfilepath):
        print('import vocab from file')
        vocab_list = get_list_from_file(vocabfilepath)
    else:
        print('build new vocab')
        vocab_list = build_vocab_list(masked_sentences, vocab_size)
        save_vocab_list_to_file(vocab_list, vocabfilepath)
    vocab, reverse_vocab = build_vocab_and_reverse_vocab(vocab_list)

    print('vectorize masked sentences')
    vectorized_sentences = [vectorize_sentence(sentence, vocab) for sentence in masked_sentences]

    print('pad sentences to common length')
    for sentence in vectorized_sentences:
        sentence += [PAD_ID] * (max_sentence_length - len(sentence))

    vectorized_sentences = np.array(vectorized_sentences)

    print('PROCESS RELATIONS')
    print('make vocab for relations')
    rel_vocabfilename = 'relations_vocab_%i.txt' % rel_vocab_size
    rel_vocabfilepath = datafolder + rel_vocabfilename
    if os.path.isfile(rel_vocabfilepath):
        print('import relations vocab from file')
        rel_vocab_list = get_list_from_file(rel_vocabfilepath)
    else:
        print('build new relations vocab')
        rel_counter = Counter(relations)
        rel_vocab_list = [_UNK] + sorted(rel_counter, key=rel_counter.get, reverse=True)
        rel_vocab_list = rel_vocab_list[:rel_vocab_size]
        save_vocab_list_to_file(rel_vocab_list, rel_vocabfilepath)
    rel_vocab, rel_reverse_vocab = build_vocab_and_reverse_vocab(rel_vocab_list)

    print('vectorize relations')
    vectorized_relations = [rel_vocab.get(relation, UNK_ID) for relation in relations]
    vectorized_relations = np.array(vectorized_relations)

    # make relations onehot
    onehot_relations = dense_to_one_hot(vectorized_relations, num_classes=rel_vocab_size)

    return onehot_relations, vectorized_sentences


def read_data(filename):
    # Open file and read in lines
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Drop trailing new line indicator '\n'
    lines = map(lambda x: x[:-3], lines)

    # filter out '#Document' lines
    lines = filter(lambda x: not x.startswith('#Document'), lines)

    # extract elements we need
    labels = map(lambda x: x.split('\t')[0], lines)
    entAs = map(lambda x: x.split('\t')[1], lines)
    entBs = map(lambda x: x.split('\t')[2], lines)
    relations = extract_column_by_prefix(lines, 'REL$')
    sentences = extract_column_by_prefix(lines, 'sen#')

    return labels, relations, sentences, entAs, entBs


def find_first_str_starting_with_prefix(list_of_strings, prefix):
    # return first instance of a string prefixed by prefix from a list of strings, or None.
    # matching string is returned with prefix removed
    return next((i[len(prefix):] for i in list_of_strings if i.startswith(prefix)), None)


def extract_column_by_prefix(list_of_tab_separated_values, prefix):
    # parse list of strings where each string consists of tab separated values
    # return a list of values corresponding to values labelled with supplied prefix
    # does not require values to be in consistent order or to have same entries
    # returns None for any string where prefix is not found.
    # returns first instance of prefix only for each string.
    return map(lambda x: find_first_str_starting_with_prefix(x.split('\t'), prefix), list_of_tab_separated_values)


def mask_entities_in_sentence(sen, entA, entB):
    return sen.replace(entA, _ENTA).replace(entB, _ENTB)


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


def build_vocab_list(sentences, vocab_size):
    vocab = {}
    counter = 0
    for sentence in sentences:
        counter += 1
        if counter % 100000 == 0:
            print("  processing line %d" % counter)
        tokens = basic_tokenizer(sentence)
        for w in tokens:
            # swap digits for 0
            word = re.sub(_DIGIT_RE, "0", w)
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)

    # crop to vocab_size
    if len(vocab_list) > vocab_size:
        vocab_list = vocab_list[:vocab_size]
    return vocab_list


def save_vocab_list_to_file(vocab_list, vocabfilepath):
    with open(vocabfilepath, 'w') as f:
        for line in vocab_list:
            f.write('%s\n' % line)


def build_vocab_and_reverse_vocab(vocab_list):
    # expects vocab_list sorted in descending word frequency order (special symbols first)
    # build vocab dict and reverse vocab list
    reverse_vocab = vocab_list
    vocab = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
    return vocab, reverse_vocab


def vectorize_sentence(sentence, vocab):
    words = basic_tokenizer(sentence)
    vectorized_sentence = []

    for w in words:
        # swap digits for 0
        word = re.sub(_DIGIT_RE, "0", w)
        word_ID = vocab.get(word, UNK_ID)
        vectorized_sentence.append(word_ID)

    return vectorized_sentence


def save_list_to_file(list_of_strings, filepath):
    with open(filepath, 'w') as f:
        for line in list_of_strings:
            f.write('%s\n' % line)


def get_list_from_file(filepath):
    with open(filepath, mode="r") as f:
        list_of_strings = [line.strip() for line in f]
    return list_of_strings


def dense_to_one_hot(dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_examples = len(dense)
    index_offset = np.arange(num_examples) * num_classes
    one_hot = np.zeros((num_examples, num_classes), dtype=np.float32)
    one_hot.flat[index_offset + dense] = 1.
    return one_hot


if __name__ == "__main__":
    main()
