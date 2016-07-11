from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np



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


def save_list_to_file(list_of_strings, filepath):
    with open(filepath, 'w') as f:
        for line in list_of_strings:
            f.write('%s\n' % line)


def get_list_from_file(filepath):
    with open(filepath, mode="r") as f:
        list_of_strings = [line.strip() for line in f]
    return list_of_strings


def save_vocab_list_to_file(vocab_list, vocabfilepath):
    with open(vocabfilepath, 'w') as f:
        for line in vocab_list:
            f.write('%s\n' % line)


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
