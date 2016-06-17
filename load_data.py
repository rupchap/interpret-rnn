
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import ast
import numpy as np
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


def main():
    datasets = read_data_sets()
    sen, rel = datasets.train.next_batch(10)
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


def read_data_sets(datafolder='/home/rupchap/data/NYT/',
                   vocab_size=10000, rel_vocab_size=25,
                   validation_size=5000, test_size=500):
    class DataSets(object):
        pass
    data_sets = DataSets()

    # check if pickled data already exist
    picklefilepath = datafolder + 'data_%i_%i.pkl' % (vocab_size, rel_vocab_size)
    if os.path.isfile(picklefilepath):
        print('load pre-processed data')
        with open(picklefilepath, 'r') as f:
            data = pickle.load(f)
        relations = np.array(data['relations'])
        sentences = np.array(data['sentences'])
    else:
        print('no pre-processed datafiles found: rebuild them')
        labels, relations, sentences = process_data(datafolder, vocab_size, rel_vocab_size)

        # filter out 'UNLABELED' cases
        valid_examples = [label != 'UNLABELED' for label in labels]
        labels = [x for (x, v) in zip(labels, valid_examples) if v]
        relations = np.array([x for (x, v) in zip(relations, valid_examples) if v])
        sentences = np.array([x for (x, v) in zip(sentences, valid_examples) if v])

        print('pickle for reuse later')
        data = {'labels': labels, 'relations': relations, 'sentences': sentences}
        with open(picklefilepath, 'w') as f:
            pickle.dump(data, f)

    # slice for training, validation, test datasets
    sentences_val = sentences[:validation_size]
    relations_val = relations[:validation_size]
    sentences_test = sentences[validation_size:validation_size+test_size]
    relations_test = relations[validation_size:validation_size+test_size]
    sentences_train = sentences[validation_size+test_size:]
    relations_train = relations[validation_size+test_size:]

    data_sets.train = DataSet(sentences_train, relations_train)
    data_sets.validation = DataSet(sentences_val, relations_val)
    data_sets.test = DataSet(sentences_test, relations_test)

    return data_sets


def process_data(datafolder='/home/rupchap/data/NYT/', vocab_size=10000, rel_vocab_size=25):

    print('IMPORT DATA')
    labels, relations, sentences, entAs, entBs = read_data(datafolder+sourcefilename)
    print('imported %i examples' % len(sentences))

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

    print('get vectorized masked sentences')
    vectorized_sentences_filename = 'sentences_%i.txt' % vocab_size
    vectorized_sentences_filepath = datafolder + vectorized_sentences_filename
    if os.path.isfile(vectorized_sentences_filepath):
        print('import vectorized masked sentences from file')
        vectorized_sentences_strings = get_list_from_file(vectorized_sentences_filepath)
        vectorized_sentences = [ast.literal_eval(line) for line in vectorized_sentences_strings]
    else:
        print('build new vectorized masked sentences')
        vectorized_sentences = [vectorize_sentence(sentence, vocab) for sentence in masked_sentences]
        save_list_to_file(vectorized_sentences, vectorized_sentences_filepath)

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
        # Save to file
        save_vocab_list_to_file(rel_vocab_list, rel_vocabfilepath)
    rel_vocab, rel_reverse_vocab = build_vocab_and_reverse_vocab(rel_vocab_list)

    print('vectorize relations')
    vectorized_relations = [rel_vocab.get(relation, UNK_ID) for relation in relations]
    vectorized_relations_filename = 'relations_%i.txt' % rel_vocab_size
    vectorized_relations_filepath = datafolder + vectorized_relations_filename
    save_list_to_file(vectorized_relations, vectorized_relations_filepath)

    # make relations onehot
    onehot_relations = dense_to_one_hot(vectorized_relations, num_classes=rel_vocab_size)

    print('save labels')
    labels_filepath = datafolder + 'labels.txt'
    save_list_to_file(labels, labels_filepath)

    return labels, onehot_relations, vectorized_sentences


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
    one_hot = np.zeros((num_examples, num_classes))
    one_hot.flat[index_offset + dense] = 1
    return one_hot


if __name__ == "__main__":
    main()
