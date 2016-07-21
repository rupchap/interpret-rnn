
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
from collections import Counter, defaultdict

from data_utils import *
from datasets import *

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
    max_sentence_length = 105
    max_shortsentence_length = 14
    rel_vocab_size = 8
    train_size = 0  # 0 to use all remaining data for training.
    validation_size = 5000
    test_size = 500
    srcfile = '/data/NYT/nyt-freebase.train.triples.universal.mention.txt'
    datafolder = '/data/train/'
    embedfolder = '/data/glove/'


def main():

    datasets = get_datasets(config=DataConfig())
    sen, rel = datasets.train.next_batch(5)
    print('training sample:')
    for s in sen:
        print(s)
    for r in rel:
        print(r)

    sen, rel, _ = datasets.test.next_batch(5)
    print('training sample:')
    for s in sen:
        print(s)
    for r in rel:
        print(r)


def get_datasets(config):

    # get data
    data = get_pickled_data_or_rebuild(config)
    datasets = build_datasets(data, config)

    return datasets


def get_pickled_data_or_rebuild(config=DataConfig()):

    datafolder = config.datafolder
    vocab_size = config.vocab_size
    rel_vocab_size = config.rel_vocab_size

    # if pickled data already exist then load it
    picklefilepath = datafolder + 'data_%i_%i.pkl' % (vocab_size, rel_vocab_size)
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
    srcfile = config.srcfile
    datafolder = config.datafolder

    data = read_data_from_source(filename=srcfile)
    save_data_by_field(data, destfolder=datafolder)
    build_short_sentences_file(folder=datafolder)

    data = read_data_from_individual_files(srcfolder=datafolder)
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


def read_data_from_individual_files(srcfolder):

    data = dict()
    field_list = ('sentences', 'entAs', 'entBs', 'relations', 'labels', 'shortsentences')

    for field in field_list:
        srcfile = srcfolder + field + '.txt'
        data[field] = get_list_from_file(srcfile)

    return data


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
            print("building vocab - processing line %d" % counter)
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


def stem_sentences(sentences):
    """very basic stemmer; just replace all digits with zeros and make lower case
    """
    # swap digits for 0
    sentences = [re.sub(_DIGIT_RE, "0", sentence) for sentence in sentences]
    # make lower case
    sentences = [sentence.lower() for sentence in sentences]

    return sentences


def build_short_sentences_file(folder='/data/train/'):
    # run external scala script to build short sentences file from paths file.
    cwd = os.path.dirname(os.path.realpath(__file__))
    scalawd = cwd + '/PathRenderer'
    subprocess.check_call(['scala', 'PathRenderer', folder], cwd=scalawd)


def process_data(data, config):

    vocab_size = config.vocab_size
    rel_vocab_size = config.rel_vocab_size
    max_sentence_length = config.max_sentence_length
    max_shortsentence_length = config.max_shortsentence_length

    data['masked_sentences'] = mask_entities_in_sentences(data['sentences'],
                                                          data['entAs'],
                                                          data['entBs'])

    data['stemmed_sentences'] = stem_sentences(data['masked_sentences'])

    vocab_list = build_vocab_list(data['stemmed_sentences'], vocab_size)
    vocab_filename = config.datafolder + ('vocab_%i' % vocab_size) + '.txt'
    save_list_to_file(vocab_list, vocab_filename)
    vocab, rev_vocab = build_vocab_and_reverse_vocab(vocab_list)

    data['sentence_vecs'] = vectorize_sentences(data['stemmed_sentences'], vocab)

    sentence_lengths = [len(sentence) for sentence in data['sentence_vecs']]
    data['sentence_lengths'] = np.array(sentence_lengths, dtype=np.int32)

    sentence_pad_vecs = [sentence + [EOS_ID] + [PAD_ID] * (max_sentence_length - len(sentence)) for
                         sentence in data['sentence_vecs']]
    data['sentence_pad_vecs'] = np.array(sentence_pad_vecs, dtype=np.int32)

    data['stemmed_shortsentences'] = stem_sentences(data['shortsentences'])

    data['shortsentence_vecs'] = vectorize_sentences(data['stemmed_shortsentences'], vocab)

    shortsentence_lengths = [len(sentence) for sentence in data['shortsentence_vecs']]
    data['shortsentence_lengths'] = np.array(shortsentence_lengths, dtype=np.int32)

    shortsentence_pad_vecs = [sentence + [EOS_ID] + [PAD_ID] * (max_shortsentence_length - len(sentence)) for
                              sentence in data['shortsentence_vecs']]
    data['shortsentence_pad_vecs'] = np.array(shortsentence_pad_vecs, np.int32)

    rel_counter = Counter(data['relations'])
    rel_vocab_list = [_UNK] + sorted(rel_counter, key=rel_counter.get, reverse=True)
    rel_vocab_list = rel_vocab_list[:rel_vocab_size]
    rel_vocab_filename = config.datafolder + ('rel_vocab_%i' % vocab_size) + '.txt'
    save_list_to_file(vocab_list, rel_vocab_filename)

    rel_vocab, rel_reverse_vocab = build_vocab_and_reverse_vocab(rel_vocab_list)

    relation_vecs = [rel_vocab.get(relation, UNK_ID) for relation in data['relations']]
    data['relation_vecs'] = np.array(relation_vecs, np.int32)

    return data


def print_sample_data(data):
    for key in data.keys():
        print(key + ' ' + data[key][0].tostr())


if __name__ == "__main__":
    main()