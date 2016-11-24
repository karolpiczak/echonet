#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from echonet.datasets.esc_original import OriginalESC


@pytest.fixture(scope='module')
def esc50():
    TRAIN_FOLDS = [2, 3, 4]
    VALIDATION_FOLDS = [5]
    TEST_FOLDS = [1]

    return OriginalESC('data/ESC-50/', 'data/.ESC-50.cache',
                       TRAIN_FOLDS, VALIDATION_FOLDS, TEST_FOLDS)


def test_input_shape(esc50):
    assert esc50.input_shape == (2, 60, 101)
