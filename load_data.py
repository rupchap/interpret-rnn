
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import numpy as np
import pandas as pd
import csv
import cPickle as pickle
import subprocess
from collections import Counter

from datasets import build_datasets


# regex for digits
_DIGIT_RE = re.compile(r"\d")
_WORD_SPLIT = ' '

# Special vocab symbols
_PAD = '_pad'
_UNK = '_unk'
_GO = '_go'
_EOS = '_eos'
_START_VOCAB = [_PAD, _UNK, _GO, _EOS]

PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3

# Symbols for masked entities
_ENTA = '_enta'
_ENTB = '_entb'


class DataConfig(object):
    vocab_size = 10000
    embed_size = 200    # 50, 100, 200 or 300 to match glove embeddings
    max_sentence_length = 106
    max_shortsentence_length = 15
    rel_vocab_size = 8
    train_size = 0  # 0 to use all remaining data for training.
    validation_size = 5000
    test_size = 500
    srcfile = '/data/NYT/nyt-freebase.train.triples.universal.mention.txt'
    datafolder = '/data/train/'
    embedfolder = '/data/glove/'


def main():

    datasets = get_datasets(config=DataConfig())
    data_train = datasets.train.next_batch(5)
    print('training sample:')
    print(data_train)


def get_datasets(config):

    # get data
    data = get_pickled_data_or_rebuild(config)
    datasets = build_datasets(data, config)

    return datasets


def get_pickled_data_or_rebuild(config=DataConfig()):

    # if pickled data already exist then load it
    picklefilepath = config.datafolder + 'data_%i_%i.pkl' % (config.vocab_size, config.rel_vocab_size)
    if os.path.isfile(picklefilepath):
        print('load pre-processed data')
        with open(picklefilepath, 'r') as f:
            data = pickle.load(f)

    # otherwise re-process original data and pickle.
    else:
        print('no pre-processed datafiles found: rebuild them')
        data = build_data(config)

        print('pickle for reuse later')
        with open(picklefilepath, 'w') as f:
            pickle.dump(data, f)

    return data


def build_data(config=DataConfig()):

    # import data from source
    data = read_data_from_source(filename=config.srcfile)

    save_data_by_field_to_individual_files(data, folder=config.datafolder)
    build_short_sentences_file(folder=config.datafolder)  # run scala script
    data = read_data_from_individual_files(folder=config.datafolder)

    data = filter_data(data)

    data = process_data(data, config)

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

    # extract fields
    data = dict()
    data['labels'] = map(lambda x: x.split('\t')[0], lines)
    data['entAs'] = map(lambda x: x.split('\t')[1], lines)
    data['entBs'] = map(lambda x: x.split('\t')[2], lines)
    data['relations'] = extract_column_by_prefix(lines, 'REL$')
    data['sentences'] = extract_column_by_prefix(lines, 'sen#')
    data['ners'] = extract_column_by_prefix(lines, 'ner#')
    data['paths'] = extract_column_by_prefix(lines, 'path#')
    data['pos'] = extract_column_by_prefix(lines, 'pos#')
    data['lc'] = extract_column_by_prefix(lines, 'lc#')
    data['rc'] = extract_column_by_prefix(lines, 'rc#')
    data['lex'] = extract_column_by_prefix(lines, 'lex#')
    data['trigger'] = extract_column_by_prefix(lines, 'trigger#')

    return data


def save_data_by_field_to_individual_files(data, folder):
    """
    create a text file for each list in data dictionary
    :param data: {fieldname: list}
    :param folder: folder in which datafiles to be saved
    :return: -
    """

    # create destination folder if doesn't exist.
    try:
        os.makedirs(folder)
    except OSError:
        if not os.path.isdir(folder):
            raise

    # save each list in dict as a separate file
    for key in data.keys():
        filepath = folder + key + '.txt'
        print('save %s as %s' % (key, filepath))
        save_list_to_file(data[key], filepath)


def build_short_sentences_file(folder='/data/train/'):
    # run external scala script to build short sentences file from paths file.
    cwd = os.path.dirname(os.path.realpath(__file__))
    scalawd = cwd + '/PathRenderer'
    subprocess.check_call(['scala', 'PathRenderer', folder], cwd=scalawd)
    print('save shortsentences as %s/shortsentences.txt' % folder)


def read_data_from_individual_files(folder):

    data = dict()
    field_list = ('sentences', 'entAs', 'entBs', 'relations', 'labels', 'shortsentences')

    for field in field_list:
        filepath = folder + field + '.txt'
        data[field] = get_list_from_file(filepath)

    return data


def filter_data(data, keep_unlabeled=False, keep_negative=True, equal_posneg=True):

    # use pandas
    data_df = pd.DataFrame(data)

    # count occurrences of 'POSITIVE', 'NEGATIVE' and 'UNLABELED' records in original data
    counts_by_label = data_df.groupby(['labels']).size()
    print('Original data:')
    print(counts_by_label)
    pos_count, neg_count, unl_count = counts_by_label[['POSITIVE', 'NEGATIVE', 'UNLABELED']]

    # Filter data to required number of cases of different label types
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

    # count occurrences of 'POSITIVE', 'NEGATIVE' and 'UNLABELED' records in filtered data
    counts_by_label = data_filtered_df.groupby(['labels']).size()
    print('Filtered data:')
    print(counts_by_label)

    # return data as dict
    data_filtered = data_filtered_df.to_dict(orient='list')

    return data_filtered


def process_data(data, config):
    
    data['masked_sentences'] = mask_entities_in_sentences(data['sentences'], data['entAs'], data['entBs'])

    data['stemmed_sentences'] = stem_sentences(data['masked_sentences'])

    vocab, rev_vocab = build_vocab(data['stemmed_sentences'], config.vocab_size)

    vocab_filename = config.datafolder + ('vocab_%i' % config.vocab_size) + '.txt'
    save_list_to_file(rev_vocab, vocab_filename)

    data['sentence_vecs'] = [vectorize_sentence(sentence, vocab, addGo=False, addEOS=True)
                             for sentence in data['stemmed_sentences']]

    # get sentence lengths - as np.array
    sentence_lengths = [len(sentence) for sentence in data['sentence_vecs']]
    data['sentence_lengths'] = np.array(sentence_lengths, dtype=np.int32)

    # pad to max sentence length - as np.array
    sentence_pad_vecs = [sentence + [PAD_ID] * (config.max_sentence_length - len(sentence)) for
                         sentence in data['sentence_vecs']]
    data['sentence_pad_vecs'] = np.array(sentence_pad_vecs, dtype=np.int32)

    data['stemmed_shortsentences'] = stem_sentences(data['shortsentences'])

    # vectorise short sentences - use same vocab as for long sentences
    data['shortsentence_vecs'] = [vectorize_sentence(sentence, vocab, addGo=True, addEOS=True)
                                  for sentence in data['stemmed_shortsentences']]

    # get short sentence lengths - as np.array
    shortsentence_lengths = [len(sentence) for sentence in data['shortsentence_vecs']]
    data['shortsentence_lengths'] = np.array(shortsentence_lengths, dtype=np.int32)

    # pad to max short sentence length - as np.array
    shortsentence_pad_vecs = [sentence + [PAD_ID] * (config.max_shortsentence_length - len(sentence)) for
                              sentence in data['shortsentence_vecs']]
    data['shortsentence_pad_vecs'] = np.array(shortsentence_pad_vecs, np.int32)

    # Create shortsentence_weights to be 1.0 for all tokens, except 0.0 for padding.
    shortsentence_weights = np.ones_like(shortsentence_pad_vecs, dtype=np.float32)
    shortsentence_weights = shortsentence_weights[shortsentence_pad_vecs == PAD_ID] - 1.
    data['shortsentence_weights'] = shortsentence_weights

    # todo: refactor to use same vocab function as for sentences
    rel_counter = Counter(data['relations'])
    rel_vocab_list = [_UNK] + sorted(rel_counter, key=rel_counter.get, reverse=True)
    rel_vocab_list = rel_vocab_list[:config.rel_vocab_size]
    rel_vocab_filename = config.datafolder + ('rel_vocab_%i' % config.rel_vocab_size) + '.txt'
    save_list_to_file(rel_vocab_list, rel_vocab_filename)

    rel_vocab = dict([(x, y) for (y, x) in enumerate(rel_vocab_list)])

    relation_vecs = [rel_vocab.get(relation, UNK_ID) for relation in data['relations']]
    data['relation_vecs'] = np.array(relation_vecs, np.int32)

    return data


def mask_entities_in_sentences(sentences, entAs, entBs):

    masked_sentences = [sentence.replace(entA, _ENTA).replace(entB, _ENTB)
                        for sentence, entA, entB in zip(sentences, entAs, entBs)]

    return masked_sentences


def stem_sentences(sentences):
    """very basic stemmer; just replace all digits with zeros and make lower case
    """
    # swap digits for 0
    sentences = [re.sub(_DIGIT_RE, "0", sentence) for sentence in sentences]
    # make lower case
    sentences = [sentence.lower() for sentence in sentences]

    return sentences


def build_vocab(sentences, vocab_size):
    vocab_counter = {}
    counter = 0
    for sentence in sentences:
        counter += 1
        if counter % 50000 == 0:
            print("building vocab - processing line %d" % counter)
        words = basic_tokenizer(sentence)
        for word in words:
            if word in vocab_counter:
                vocab_counter[word] += 1
            else:
                vocab_counter[word] = 1
    reverse_vocab = _START_VOCAB + sorted(vocab_counter, key=vocab_counter.get, reverse=True)

    # crop to vocab_size
    if len(reverse_vocab) > vocab_size:
        reverse_vocab = reverse_vocab[:vocab_size]

    # build vocab dict
    vocab = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])

    return vocab, reverse_vocab


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [word for word in words if word]


def vectorize_sentence(sentence, vocab, addGo=False, addEOS=True):

    vectorized_sentence = []

    words = basic_tokenizer(sentence)

    if addGo:
        vectorized_sentence.append(GO_ID)

    for word in words:
        word_ID = vocab.get(word, UNK_ID)
        vectorized_sentence.append(word_ID)

    if addEOS:
        vectorized_sentence.append(EOS_ID)

    return vectorized_sentence


def shorten_sentence(sen, entA, entB):
    start = sen.find(entA) + len(entA) + 1
    end = sen.find(entB) - 1
    return sen[start:end]


def build_initial_embedding(config):

    embed_folder = config.embedfolder
    vocab_folder = config.datafolder
    embed_size = config.embed_size
    vocab_size = config.vocab_size

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


def print_sample_data(data):
    for key in data.keys():
        print(key + ' ' + data[key][0].tostr())


def split_arrays(arr, train_size, validation_size, test_size):
    assert validation_size + test_size + train_size <= arr.shape[0]

    arr_val, arr = np.split(arr, [validation_size])
    arr_test, arr = np.split(arr, [test_size])
    arr_train, arr = np.split(arr, [train_size])

    return arr_train, arr_val, arr_test


def dense_to_one_hot(dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_examples = len(dense)
    index_offset = np.arange(num_examples) * num_classes
    one_hot = np.zeros((num_examples, num_classes), dtype=np.float32)
    one_hot.flat[index_offset + dense] = 1.
    return one_hot


def get_list_from_file(filepath):
    with open(filepath, mode="r") as f:
        list_of_strings = [line.strip() for line in f]
    return list_of_strings


def extract_column_by_prefix(list_of_tab_separated_values, prefix):
    # parse list of strings where each string consists of tab separated values
    # return a list of values corresponding to values labelled with supplied prefix
    # does not require values to be in consistent order or to have same entries
    # returns None for any string where prefix is not found.
    # returns first instance of prefix only for each string.
    return map(lambda x: find_first_str_starting_with_prefix(x.split('\t'), prefix), list_of_tab_separated_values)


def find_first_str_starting_with_prefix(list_of_strings, prefix):
    """ return first instance of a string prefixed by prefix from a list of strings, or None.
    :param list_of_strings: list of strings to search
    :param prefix: str prefix appearing at start of required string
    :return: string matching prefix pattern (with prefix removed)
    """
    return next((i[len(prefix):] for i in list_of_strings if i.startswith(prefix)), None)


def save_list_to_file(list_of_strings, filepath):
    with open(filepath, 'w') as f:
        for string in list_of_strings:
            f.write('%s\n' % string)


if __name__ == "__main__":
    main()
