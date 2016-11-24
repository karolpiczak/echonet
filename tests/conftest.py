#!/usr/bin/env python
# -*- coding: utf-8 -*-


def pytest_addoption(parser):
    parser.addoption('--GPU', action='store_true', help='Use GPU for testing')


def pytest_generate_tests(metafunc):
    if 'device' in metafunc.fixturenames:
        if metafunc.config.option.GPU:
            metafunc.parametrize('device', ['gpu0'])
        else:
            metafunc.parametrize('device', ['cpu'])