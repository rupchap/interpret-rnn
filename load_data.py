
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import numpy as np
import pandas as pd
import csv
import cPickle as pickle
from collections import Counter, defaultdict


sourcefilename = 'nyt-freebase.train.triples.universal.mention.txt'

# regex for digits
_DIGIT_RE = re.compile(r"\d")
_WORD_SPLIT = ' '

# Special vocabulary symbols
_PAD = '_pad'
_UNK = '_unk'
_GO = '_go'
_EOS = '_eos'
_START_VOCAB = [ _PAD, _UNK, _GO, _EOS]


PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3

# Symbols for masked entities
_ENTA = '_enta'
_ENTB = '_entb'


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

    def __init__(self, data):
        """Construct a DataSet.
        """

        self._sentences = data['sentences']
        self._relations = data['relations']
        self._lengths = data['lengths']
        self._num_examples = self._sentences.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0

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
            self._lengths = self._lengths[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch

        # return batch
        return self._sentences[start:end], self._relations[start:end], self._lengths[start:end]

    def get_all(self):
        """Return all examples from this data set."""
        return self._sentences, self._relations, self._lengths

    @property
    def sentences(self):
        return self._sentences

    @property
    def relations(self):
        return self._relations

    @property
    def lengths(self):
        return self._lengths

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed


def get_data(datafolder='/data/train/', vocab_size=10000, rel_vocab_size=10):
    # if pickled data already exist then load it
    picklefilepath = datafolder + 'data_%i_%i.pkl' % (vocab_size, rel_vocab_size)
    if os.path.isfile(picklefilepath):
        print('load pre-processed data')
        with open(picklefilepath, 'r') as f:
            data = pickle.load(f)
    # otherwise re-process original data and pickle.
    else:
        print('no pre-processed datafiles found: rebuild them')
        data = build_data(datafolder, vocab_size, rel_vocab_size)

        print('pickle for reuse later')
        with open(picklefilepath, 'w') as f:
            pickle.dump(data, f)

    return data


def build_data_sets(data, validation_size=5000, test_size=500, train_size=0):
    class DataSets(object):
        pass
    data_sets = DataSets()

    relations = data['relations']
    sentences = data['sentences']
    lengths = data['lengths']

    # slice for training, validation, test datasets
    num_examples = sentences.shape[0]
    if not train_size:
        train_size = num_examples - validation_size - test_size

    data_train, data_rem = split_data(data, train_size)
    data_val, data_rem = split_data(data_rem, validation_size)
    data_test, _ = split_data(data_rem, test_size)

    data_sets.train = DataSet(data_train)
    data_sets.validation = DataSet(data_val)
    data_sets.test = DataSet(data_test)

    return data_sets

def split_data(data, split_size):
    """ takes a dict of arrays and splits into two dicts, one of size split_size other of remainder
    :param data: dict of numpy arrays, all of same length in 1st axis
    :param split_size: int number of rows to be included in first split
    :return: data_top [dict of size split_size], data_btm [dict of remainder]
    """
    data_top = dict()
    data_btm = dict()

    for key in data.keys:
        val_top, val_btm = np.split(data[key], split_size)
        data_top[key] = val_top
        data_btm[key] = val_btm

    return data_top, data_btm


def filter_data(data, keep_unlabeled=False, keep_negative=True, equal_posneg=True):

    data_df = pd.DataFrame(data)

    counts_by_label = data_df.groupby(['labels']).size()
    print('Original data:')
    print(counts_by_label)
    pos_count, neg_count, unl_count = counts_by_label[['POSITIVE', 'NEGATIVE', 'UNLABELED']]

    # Filter dataset to required number of cases of different label types
    pos_req = pos_count
    if keep_negative:
        if equal_posneg:
            neg_req = min(pos_req, neg_count)
            pos_req = min(neg_req, pos_req)
        else:
            neg_req = neg_count
    else:
        neg_req = 0

    if keep_unlabeled:
        unl_req = unl_count
    else:
        unl_req = 0

    pos = data_df[data_df['labels'] == 'POSITIVE'].sample(n=pos_req)
    neg = data_df[data_df['labels'] == 'NEGATIVE'].sample(n=neg_req)
    unl = data_df[data_df['labels'] == 'UNLABELED'].sample(n=unl_req)
    data_filtered_df = pd.concat([pos, neg, unl])
    # shuffle rows
    data_filtered_df = data_filtered_df.sample(frac=1)

    counts_by_label = data_filtered_df.groupby(['labels']).size()
    print('Filtered data:')
    print(counts_by_label)

    data_filtered = data_filtered_df.to_dict(orient='list')

    return data_filtered


def process_data(data, vocab_size=10000, rel_vocab_size=25):

    sentences = data['sentences']
    masked_sentences = data['masked_sentences']

    print('build vocab')
    vocabfilename = 'vocab_%i.txt' % vocab_size
    vocabfilepath = datafolder + vocabfilename
    vocab_list = build_vocab_list(masked_sentences, vocab_size)
    save_vocab_list_to_file(vocab_list, vocabfilepath)
    vocab, reverse_vocab = build_vocab_and_reverse_vocab(vocab_list)

    print('vectorize masked sentences')
    vectorized_sentences = [vectorize_sentence(sentence, vocab) for sentence in masked_sentences]

    print('count sentence lengths')
    sentence_lengths = [len(sentence) for sentence in vectorized_sentences]
    sentence_lengths = np.array(sentence_lengths)

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

    data = {'relations': onehot_relations, 'sentences': vectorized_sentences, 'lengths': sentence_lengths}

    return data


def read_data_from_individual_files(srcfolder):

    data = dict()
    field_list = ('sentences', 'entAs', 'entBs', 'relations', 'labels', 'shortsentences')

    for field in field_list:
        srcfile = srcfolder + field + '.txt'
        data[field] = get_list_from_file(srcfile)

    return data

def read_data_from_source(filename):
    """ reads in data from specified file, extracts fields and returns as a dict.
    :param filename: source text file NYT-freebase corpus
    :return: data dict.
    """
    # Open file and read in lines
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Drop trailing new line indicator '\n'
    lines = map(lambda x: x[:-3], lines)

    # filter out '#Document' lines
    lines = filter(lambda x: not x.startswith('#Document'), lines)

    # extract elements we need

    return data


def save_data_by_field(data, destfolder):
    """
    create a text file for each list in data dictionary
    :param data: {fieldname: list}
    :param destfolder: folder in which datafiles to be saved
    :return: -
    """

    # create destination folder if doesn't exist.
    try:
        os.makedirs(destfolder)
    except OSError:
        if not os.path.isdir(destfolder):
            raise

    for key in data.keys():
        destfile = destfolder + key + '.txt'
        print('save %s as %s' % (key, destfile))
        save_list_to_file(data[key], destfile)


def find_first_str_starting_with_prefix(list_of_strings, prefix):
    """ return first instance of a string prefixed by prefix from a list of strings, or None.
    :param list_of_strings: list of strings to search
    :param prefix: str prefix appearing at start of required string
    :return: string matching prefix pattern (with prefix removed)
    """
    return next((i[len(prefix):] for i in list_of_strings if i.startswith(prefix)), None)


def extract_column_by_prefix(list_of_tab_separated_values, prefix):
    # parse list of strings where each string consists of tab separated values
    # return a list of values corresponding to values labelled with supplied prefix
    # does not require values to be in consistent order or to have same entries
    # returns None for any string where prefix is not found.
    # returns first instance of prefix only for each string.
    return map(lambda x: find_first_str_starting_with_prefix(x.split('\t'), prefix), list_of_tab_separated_values)


def mask_entities_in_sentences(sentences, entAs, entBs):

    masked_sentences = [sentence.replace(entA, '_ENTA').replace(entB, '_ENTB')
                        for sentence, entA, entB in zip(sentences, entAs, entBs)]

    return masked_sentences


def shorten_sentence(sen, entA, entB):
    start = sen.find(entA) + len(entA) + 1
    end = sen.find(entB) - 1
    return sen[start:end]


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
        words = basic_tokenizer(sentence)
        for word in words:
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


def vectorize_sentences(sentences, vocab):
    vectorized_sentences = [vectorize_sentence(sentence, vocab) for sentence in sentences]
    return vectorized_sentences


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


def split_arrays(arr, train_size, validation_size, test_size):
    assert validation_size + test_size + train_size <= arr.shape[0]

    arr_val, arr = np.split(arr, [validation_size])
    arr_test, arr = np.split(arr, [test_size])
    arr_train, arr = np.split(arr, [train_size])

    return arr_train, arr_val, arr_test


def build_initial_embedding(embed_folder, vocab_folder, embed_size, vocab_size):
    embedfilepath = embed_folder + 'glove.6B.%id.txt' % embed_size
    vocabfilepath = vocab_folder + 'vocab_%i.txt' % vocab_size

    # read pre-cooked embeddings and vocab list
    embed = pd.read_csv(embedfilepath, delim_whitespace=True, quoting=csv.QUOTE_NONE, header=None)
    vocab = pd.read_csv(vocabfilepath, delim_whitespace=True, quoting=csv.QUOTE_NONE, header=None)

    # merge where vocab words exist in embeddings
    vocab_embed = vocab.merge(embed, how='left', on=0)

    embed_count = embed.shape[0]
    vocab_count = vocab.shape[0]
    notfound_embed_count = vocab_embed[1].isnull().sum()
    print('vocab size: %i, pre-trained embeddings: %i, vocab not found in pretrained: %i (%f)' %
          (vocab_count, embed_count, notfound_embed_count, 1. * notfound_embed_count / vocab_count))

    # generate random embeddings to patch missing values [assume mean 0, match stdev of full embed data]
    embed_stdev = embed.drop(0, axis=1).std(axis=0).mean()
    embed_rand = pd.DataFrame(embed_stdev * np.random.randn(vocab_embed.shape[0], vocab_embed.shape[1]))

    # patch missing with random values [drop 1st col of word strings]
    vocab_embed = vocab_embed.drop(0, axis=1)
    embed_rand = embed_rand.drop(0, axis=1)
    vocab_embed = vocab_embed.fillna(embed_rand)

    embedding = np.array(vocab_embed.values, dtype=np.float32)

    return embedding


def add_parsed_sentences_to_data(data):

    sentences = data['sentences']
    entAs = data['entAs']
    entBs = data['entBs']

    # mask entities
    sentences = mask_entities_in_sentences(sentences, entAs, entBs)

    # stem sentences
    sentences = stem_sentences(sentences)

    data['parsed_sentences'] = sentences

    return data


def stem_sentences(sentences):
    """very basic stemmer; just replace all digits with zeros and make lower case
    """
    # swap digits for 0
    sentences = [re.sub(_DIGIT_RE, "0", sentence) for sentence in sentences]
    # make lower case
    sentences = [sentence.lower() for sentence in sentences]

    return sentences


def extract_source_data(srcfolder='/data/NYT/nyt-freebase.train.triples.universal.mention.txt',
                        dstfolder='/data/train/'):

    data = read_data_from_source(srcfolder)
    save_data_by_field(data, dstfolder)

    print('data from %s saved by field in %s' % (srcfolder, dstfolder))


def build_data(srcfolder='/data/train', vocab_size=10000, rel_vocab_size=10,
               max_sentence_length=104, max_shortsentence_length=25):

    data = read_data_from_individual_files('/data/train/')

    data = filter_data(data)

    data['masked_sentences'] = mask_entities_in_sentences(data['sentences'],
                                                          data['entAs'],
                                                          data['entBs'])

    data['stemmed_sentences'] = stem_sentences(data['masked_sentences'])

    vocab_list = build_vocab_list(data['stemmed_sentences'], vocab_size)
    vocab, rev_vocab = build_vocab_and_reverse_vocab(vocab_list)

    data['sentence_vecs'] = vectorize_sentences(data['stemmed_sentences'], vocab)

    data['sentence_lengths'] = [len(sentence) for sentence in data['sentence_vecs']]

    sentence_pad_vecs = [sentence + [PAD_ID] * (max_sentence_length - len(sentence)) for
                         sentence in data['sentence_vecs']]
    data['sentence_pad_vecs'] = np.array(sentence_pad_vecs, dtype=np.int32)

    data['stemmed_shortsentences'] = stem_sentences(data['shortsentences'])

    data['shortsentence_vecs'] = vectorize_sentences(data['stemmed_shortsentences'], vocab)

    data['shortsentence_lengths'] = [len(sentence) for sentence in data['shortsentence_vecs']]

    shortsentence_pad_vecs = [sentence + [PAD_ID] * (max_shortsentence_length - len(sentence)) for
                              sentence in data['shortsentence_vecs']]
    data['shortsentence_pad_vecs'] = np.array(shortsentence_pad_vecs, np.int32)

    rel_counter = Counter(data['relations'])
    rel_vocab_list = [_UNK] + sorted(rel_counter, key=rel_counter.get, reverse=True)
    rel_vocab_list = rel_vocab_list[:rel_vocab_size]
    rel_vocab, rel_reverse_vocab = build_vocab_and_reverse_vocab(rel_vocab_list)

    relation_vecs = [rel_vocab.get(relation, UNK_ID) for relation in data['relations']]
    data['relation_vecs'] = np.array(relation_vecs, np.int32)

    return data

if __name__ == "__main__":

    data = build_data(srcfolder='/data/train', vocab_size=10000, rel_vocab_size=10)

    print(data['sentences'][5])
    print(data['masked_sentences'][5])
    print(data['stemmed_sentences'][5])
    print(data['sentence_vecs'][5])
    print(data['sentence_pad_vecs'][5])

    print(data['shortsentences'][5])
    print(data['stemmed_shortsentences'][5])
    print(data['shortsentence_vecs'][5])
    print(data['shortsentence_pad_vecs'][5])

    print(data['relations'][5])
    print(data['relation_vecs'][5])

    print(data['labels'][5])

    # main()
