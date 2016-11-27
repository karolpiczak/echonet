#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from echonet.datasets.esc_original import OriginalESC


TRAIN_FOLDS = [2, 3, 4]
VALIDATION_FOLDS = [5]
TEST_FOLDS = [1]


@pytest.fixture(scope='module')
def esc10():
    return OriginalESC('data/ESC-50/', 'data/.ESC-50.cache',
                       TRAIN_FOLDS, VALIDATION_FOLDS, TEST_FOLDS, True)


@pytest.fixture(scope='module')
def esc50():
    return OriginalESC('data/ESC-50/', 'data/.ESC-50.cache',
                       TRAIN_FOLDS, VALIDATION_FOLDS, TEST_FOLDS)


def test_input_shape(esc50):
    assert esc50.input_shape == (2, 60, 101)


def test_esc10_size(esc10):
    assert len(esc10.meta) == 400
    assert esc10.train_size == 240
    assert esc10.test_size == 80
    assert esc10.test_segments == 480
    assert esc10.validation_size == 80
    assert esc10.validation_segments == 480


def test_esc50_size(esc50):
    assert len(esc50.meta) == 2000
    assert esc50.train_size == 1200
    assert esc50.test_size == 400
    assert esc50.test_segments == 2400
    assert esc50.validation_size == 400
    assert esc50.validation_segments == 2400


def test_esc10_batches(esc10):
    ibatches = esc10.iterbatches(100)

    for i in range(20):  # iterate at least once over the dataset
        batch = next(ibatches)

    expected = np.load('tests/echonet/datasets/esc10.npz')

    assert np.allclose(batch[0][0:5], expected['X'])
    assert np.allclose(batch[1][0:5], expected['y'])


def test_esc50_batches(esc50):
    ibatches = esc50.iterbatches(100)

    for i in range(20):  # iterate at least once over the dataset
        batch = next(ibatches)

    expected = np.load('tests/echonet/datasets/esc50.npz')

    assert np.allclose(batch[0][0:5], expected['X'])
    assert np.allclose(batch[1][0:5], expected['y'])
