#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test generic convnet model on ESC-50."""

import os
import re
import shutil

import numpy as np
import pytest


@pytest.mark.timeout(2400)
def test_results_reproducibility(capsys, device):
    np.random.seed(20161013)
    THEANO_FLAGS = ('device={},'
                    'floatX=float32,'
                    'dnn.conv.algo_bwd_filter=deterministic,'
                    'dnn.conv.algo_bwd_data=deterministic').format(device)
    os.environ['THEANO_FLAGS'] = THEANO_FLAGS
    os.environ['KERAS_BACKEND'] = 'theano'

    import keras
    keras.backend.set_image_dim_ordering('th')
    from keras.layers.convolutional import Convolution2D as Conv
    from keras.layers.convolutional import MaxPooling2D as Pool
    from keras.layers.advanced_activations import LeakyReLU
    from keras.layers.core import Activation, Dense, Dropout, Flatten

    from echonet.models import EchoNet
    from echonet.datasets.esc import ESC

    TRAIN_FOLDS = [2, 3, 4]
    VALIDATION_FOLDS = [5]
    TEST_FOLDS = [1]

    TEST_DIR = 'data/.ESC-50.test.cache'
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    os.mkdir(TEST_DIR)

    esc50 = ESC('data/ESC-50/', TEST_DIR, TRAIN_FOLDS, VALIDATION_FOLDS, TEST_FOLDS)

    input_shape = esc50.input_shape
    L2 = keras.regularizers.l2

    l = []

    l += [Conv(80, 57, 6, init='he_uniform', W_regularizer=L2(0.001), input_shape=input_shape)]
    l += [LeakyReLU()]
    l += [Pool((4, 6), (1, 3))]

    l += [Conv(160, 1, 2, W_regularizer=L2(0.001), init='he_uniform')]
    l += [LeakyReLU()]
    l += [Pool((1, 2), (1, 2))]

    l += [Conv(240, 1, 2, W_regularizer=L2(0.001), init='he_uniform')]
    l += [LeakyReLU()]
    l += [Pool((1, 2), (1, 2))]

    l += [Conv(320, 1, 2, W_regularizer=L2(0.001), init='he_uniform')]
    l += [LeakyReLU()]
    l += [Pool((1, 2), (1, 2))]

    l += [Flatten()]

    l += [Dropout(0.5)]

    l += [Dense(esc50.class_count, W_regularizer=L2(0.001), init='he_uniform')]
    l += [Activation('softmax')]

    optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)

    model = EchoNet('test_esc50', layers=l, optimizer=optimizer)

    # Initialize biases of the first convolutional layer
    conv_weights = model.layers[0].get_weights()
    conv_weights[1][:] = 1
    model.layers[0].set_weights(conv_weights)

    print(model)

    epoch_size = esc50.train_size * 10
    model.fit(esc50, batch_size=100, epochs=3, epoch_size=epoch_size)

    if device == 'gpu0':
        expected_results = [
            r'Epoch:   0 (.*) | Train:  12.00 % | Validation:  21.70 % | Test:  25.50 %',
            r'Epoch:   1 (.*) | Train:  24.20 % | Validation:  28.50 % | Test:  32.00 %',
            r'Epoch:   2 (.*) | Train:  32.10 % | Validation:  33.20 % | Test:  37.50 %',
        ]
    elif device == 'cpu':
        expected_results = [
            r'Epoch:   0 (.*) | Train:   9.90 % | Validation:  20.20 % | Test:  23.50 %',
            r'Epoch:   1 (.*) | Train:  21.20 % | Validation:  28.00 % | Test:  30.70 %',
            r'Epoch:   2 (.*) | Train:  28.00 % | Validation:  33.00 % | Test:  35.00 %',
        ]

    out, err = capsys.readouterr()

    for line in expected_results:
        assert re.search(line, out) is not None

    os.remove('test_esc50.log')


if __name__ == '__main__':
    pytest.main([__file__])
