# -*- coding: utf-8 -*-
"""

"""

from abc import ABC, abstractmethod, abstractproperty
import os
import types

import numpy as np


class Dataset(ABC):
    """

    """
    def __init__(self, data_dir, work_dir):
        self.RandomState = np.random.RandomState(seed=20161013)
        self.data_mean = None
        self.data_std = None
        self.class_count = None
        self.meta = None
        self.train_meta = None
        self.test_data = types.SimpleNamespace(X=None, y=None, meta=None)
        self.validation_data = types.SimpleNamespace(X=None, y=None, meta=None)
        self.data_dir = data_dir
        self.work_dir = work_dir

        self.data_dir = os.path.abspath(self.data_dir) + '/'
        self.work_dir = os.path.abspath(self.work_dir) + '/'

        if not os.path.isdir(self.data_dir):
            raise NotADirectoryError("{} is not a proper dataset directory.".format(self.data_dir))
        if not os.path.isdir(self.work_dir):
            raise NotADirectoryError("{} is not a proper working directory.".format(self.work_dir))

    def _iterrows(self, dataset):
        while True:
            for row in dataset.iloc[self.RandomState.permutation(len(dataset))].itertuples():
                yield row

    @abstractproperty
    def input_shape(self): pass

    @abstractmethod
    def iterbatches(self, batch_size): pass

    @abstractproperty
    def train_size(self): pass

    @abstractmethod
    def validate(self, model): pass

    @abstractproperty
    def validation_size(self): pass

    @abstractmethod
    def test(self, model): pass

    @abstractproperty
    def test_size(self): pass


