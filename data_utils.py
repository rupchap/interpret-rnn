from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd


def split_data(data, split_size):
    """ takes a dict of arrays and splits into two dicts, one of size split_size other of remainder
    :param data: dict of numpy arrays, all of same length in 1st axis
    :param split_size: int number of rows to be included in first split
    :return: data_top [dict of size split_size], data_btm [dict of remainder]
    """
    data_top = dict()
    data_btm = dict()

    for key in data.keys():
        if type(data[key]) == np.ndarray:
            val_top, val_btm = np.split(data[key], [split_size])
        else:
            val_top, val_btm = data[key][:split_size], data[key][split_size:]

        data_top[key] = val_top
        data_btm[key] = val_btm

    return data_top, data_btm


    # Shuffle the data
#
# def shuffle_data(data, split_size):
#
#     perm = np.arange(self._num_examples)
#     np.random.shuffle(perm)
#
#     self._sentences = self._sentences[perm]
#     self._relations = self._relations[perm]
#     self._lengths = self._lengths[perm]