import numpy as np
from load_data import *


class DataSet(object):

    def __init__(self, data):
        """Construct a DataSet.
        """

        self._sentences = data['sentence_pad_vecs']
        self._lengths = data['sentence_lengths']
        self._shortsentences = data['shortsentence_pad_vecs']
        self._shortlengths = data['shortsentence_lengths']
        self._relations = data['relations']
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
