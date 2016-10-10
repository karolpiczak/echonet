# -*- coding: utf-8 -*-
"""

"""

from abc import ABC, abstractmethod
import os

import numpy as np
import pydub


def _load_audio(path, sr=44100):
    audio = pydub.AudioSegment.from_file(path).set_frame_rate(sr).set_channels(1)
    return (np.fromstring(audio._data, dtype="int16") + 0.5) / (0x7FFF + 0.5)  # convert to float


def to_percentage(number):
    return int(number * 1000) / 10.0


def to_one_hot(targets, class_count):
    one_hot_enc = np.zeros((len(targets), class_count))

    for r in range(len(targets)):
        one_hot_enc[r, targets[r]] = 1

    return one_hot_enc


class Dataset(ABC):
    """

    """
    def __init__(self, data_dir, work_dir):
        self.data_mean = None
        self.data_std = None
        self.class_count = None
        self.meta = None
        self.data_dir = data_dir
        self.work_dir = work_dir

        self.data_dir = os.path.abspath(self.data_dir) + '/'
        self.work_dir = os.path.abspath(self.work_dir) + '/'

        if not os.path.isdir(self.data_dir):
            raise NotADirectoryError("{} is not a proper dataset directory.".format(self.data_dir))
        if not os.path.isdir(self.work_dir):
            raise NotADirectoryError("{} is not a proper working directory.".format(self.work_dir))

    @staticmethod
    def _loop_dataset(dataset, randomize=True):
        if randomize:
            while True:
                for row in dataset.iloc[np.random.permutation(len(dataset))].itertuples():
                    yield row
        else:
            while True:
                for row in dataset.itertuples():
                    yield row

    @abstractmethod
    def get_train_batch(self, batch_size): pass

    @abstractmethod
    def get_train_size(self): pass

    @abstractmethod
    def get_validation_batch(self, batch_size): pass

    @abstractmethod
    def get_validation_size(self): pass

    @abstractmethod
    def get_test_all(self): pass

    @abstractmethod
    def get_test_size(self): pass

    @abstractmethod
    def get_input_shape(self): pass

    @abstractmethod
    def score(self, targets, predictions, ensembling_mode=None): pass


