import numpy as np
import pandas as pd


class DataSet(object):

    def __init__(self, data):
        """Construct a DataSet.
        """

        # Select and relabel the elements of data needed for model dataset
        dataset_dict = {'sentence': 'sentence_pad_vecs',
                        'sentence_lengths': 'sentence_lengths',
                        'short': 'shortsentence_pad_vecs',
                        'short_lengths': 'shortsentence_lengths',
                        'short_weights': 'shortsentence_weights',
                        'relation': 'relation_vecs'
                        }
        self._data = {key: data[dataset_dict[key]] for key in dataset_dict}

        self._num_examples = self._data['sentence'].shape[0]

        origtext_dict = {'sentence_text': 'sentences',
                         'enta_text': 'entAs',
                         'entb_text': 'entBs',
                         'short_text': 'shortsentences',
                         'relation_text': 'relations'
                         }
        self._origtext = {key: data[origtext_dict[key]] for key in origtext_dict}

        # Track position in data
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size, include_text=False):
        """Return the next `batch_size` examples from this data set."""

        # update position in data
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            for key in self._data.keys():
                self._data[key] = self._data[key][perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch

        data_batch = {key: self._data[key][start:end] for key in self._data.keys()}

        # include original text strings if required
        if include_text:
            origtext_batch = {key: self._origtext[key][start:end] for key in self._origtext.keys()}
            data_batch.update(origtext_batch)

        return data_batch

    @property
    def data(self):
        return self._data

    @property
    def origtext(self):
        return self._origtext

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed


def build_datasets(data, config):
    class DataSets(object):
        pass
    datasets = DataSets()

    validation_size = config.validation_size
    test_size = config.test_size
    train_size = config.train_size

    # slice for training, validation, test datasets
    num_examples = len(data['sentences'])
    if not train_size:
        train_size = num_examples - validation_size - test_size

    data_train, data_rem = split_data(data, train_size)
    data_val, data_rem = split_data(data_rem, validation_size)
    data_test, _ = split_data(data_rem, test_size)

    datasets.train = DataSet(data_train)
    datasets.validation = DataSet(data_val)
    datasets.test = DataSet(data_test)

    return datasets


def split_data(data, split_size):
    """ takes a dict of arrays and splits into two dicts, one of size split_size other of remainder
    :param data: dict of numpy arrays AND/OR lists, all of same length in 1st axis
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
