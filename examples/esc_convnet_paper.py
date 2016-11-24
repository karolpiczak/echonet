#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Environmental Sound Classification with Convolutional Neural Networks.

Paper source code ported to Keras with some small adjustments.

# Reference:

- [Environmental Sound Classification with Convolutional Neural Networks -
    paper replication data](https://github.com/karoldvl/paper-2015-esc-convnet)

# Expected results on a GTX 980 Ti GPU:

```
Epoch:   0 -  47.4 s /  59.9 s | Train:   1.90 % | Validation:   2.50 % | Test:   3.00 %
Epoch:   1 -  37.0 s /  38.7 s | Train:   2.20 % | Validation:   1.50 % | Test:   2.20 %
Epoch:   2 -  46.3 s /  47.9 s | Train:   3.20 % | Validation:   4.50 % | Test:   5.00 %
Epoch:   3 -  46.4 s /  48.1 s | Train:   5.10 % | Validation:   7.00 % | Test:   8.20 %
Epoch:   4 -  46.2 s /  47.7 s | Train:   7.80 % | Validation:  10.50 % | Test:  11.70 %

[...]

Epoch: 145 -  46.8 s /  48.6 s | Train:  92.00 % | Validation:  63.50 % | Test:  66.50 %
Epoch: 146 -  47.2 s /  49.2 s | Train:  91.90 % | Validation:  62.50 % | Test:  68.50 %
Epoch: 147 -  47.2 s /  48.9 s | Train:  92.00 % | Validation:  61.70 % | Test:  66.50 %
Epoch: 148 -  47.2 s /  48.9 s | Train:  92.20 % | Validation:  63.00 % | Test:  67.00 %
Epoch: 149 -  46.2 s /  47.9 s | Train:  92.30 % | Validation:  62.20 % | Test:  67.00 %
```

# Expected results with CPU computation (AMD FX-8350):

```
Epoch:   0 - 183.7 s / 199.1 s | Train:   2.00 % | Validation:   2.00 % | Test:   2.00 %
Epoch:   1 - 170.6 s / 180.2 s | Train:   2.30 % | Validation:   3.70 % | Test:   2.50 %
Epoch:   2 - 178.8 s / 188.6 s | Train:   4.80 % | Validation:   4.50 % | Test:   4.50 %
Epoch:   3 - 180.4 s / 190.3 s | Train:   6.10 % | Validation:   5.70 % | Test:   5.50 %
Epoch:   4 - 175.3 s / 184.9 s | Train:   8.00 % | Validation:  10.70 % | Test:  10.00 %

[...]


```

"""

import argparse
import functools
import os
import sys

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--device', help='Theano device used for computations')
    args = parser.parse_args()

    RANDOM_SEED = 20161013
    np.random.seed(RANDOM_SEED)

    DEVICE = args.device if args.device else 'gpu0'
    THEANO_FLAGS = ('device={},'
                    'floatX=float32,'
                    'dnn.conv.algo_bwd_filter=deterministic,'
                    'dnn.conv.algo_bwd_data=deterministic').format(DEVICE)
    os.environ['THEANO_FLAGS'] = THEANO_FLAGS
    os.environ['KERAS_BACKEND'] = 'theano'

    sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/..')

    import keras
    keras.backend.set_image_dim_ordering('th')
    from keras.layers.convolutional import Convolution2D as Conv
    from keras.layers.convolutional import MaxPooling2D as Pool
    from keras.layers.core import Activation, Dense, Dropout, Flatten

    from echonet.models import EchoNet
    from echonet.datasets.esc_original import OriginalESC


    def uniform(scale):
        return functools.partial(keras.initializations.uniform, scale=scale)

    def normal(stdev):
        return functools.partial(keras.initializations.normal, scale=stdev)

    TRAIN_FOLDS = [2, 3, 4]
    VALIDATION_FOLDS = [5]
    TEST_FOLDS = [1]

    print('\nLoading ESC-50 dataset from ../data/ESC-50/')
    esc50 = OriginalESC('../data/ESC-50/', '../data/.ESC-50.cache', TRAIN_FOLDS, VALIDATION_FOLDS,
                        TEST_FOLDS)

    input_shape = esc50.input_shape
    L2 = keras.regularizers.l2

    layers = [
        Conv(80, 57, 6, init=uniform(0.001), W_regularizer=L2(0.001), input_shape=input_shape),
        Activation('relu'),
        Pool((4, 3), (1, 3)),
        Dropout(0.5),
        Conv(80, 1, 3, init=uniform(0.1), W_regularizer=L2(0.001)),
        Activation('relu'),
        Pool((1, 3), (1, 3)),
        Flatten(),
        Dense(5000, init=normal(0.01), W_regularizer=L2(0.001)),
        Activation('relu'),
        Dropout(0.5),
        Dense(5000, init=normal(0.01), W_regularizer=L2(0.001)),
        Activation('relu'),
        Dropout(0.5),
        Dense(esc50.class_count, init=normal(0.01), W_regularizer=L2(0.001)),
        Activation('softmax')
    ]

    optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)

    print('\nCompiling Keras model')
    model = EchoNet('esc_convnet_paper', layers=layers, optimizer=optimizer)

    # Initialize biases of the first convolutional layer
    conv_weights = model.layers[0].get_weights()
    conv_weights[1][:] = 0.1
    model.layers[0].set_weights(conv_weights)

    print('\nTraining... (batch size of 1 000 | 30 batches per epoch)')
    epoch_size = esc50.train_size * 25    # Approximation of the original paper
    model.fit(esc50, batch_size=1000, epochs=150, epoch_size=epoch_size)

    model.net.save('esc_convnet_paper.h5')
