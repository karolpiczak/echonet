#!/usr/bin/env python
# -*- coding: utf-8 -*-


def pytest_addoption(parser):
    parser.addoption('--GPU', action='store_true', help='Use GPU for testing')
    parser.addoption('--timeout', action='store', default=900, type=int, help='Testing timeout')


def pytest_generate_tests(metafunc):
    if 'device' in metafunc.fixturenames and 'timeout' in metafunc.fixturenames:
        if metafunc.config.option.GPU:
            metafunc.parametrize(['device', 'timeout'],
                                 [['gpu0', metafunc.config.option.timeout]])
        else:
            metafunc.parametrize(['device', 'timeout'],
                                 [['cpu', metafunc.config.option.timeout]])
