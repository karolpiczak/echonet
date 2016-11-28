#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import subprocess

from subprocess import PIPE, STDOUT

import pytest


@pytest.mark.timeout(2400)
def test_results_reproducibility(device):
    if device == 'gpu0':
        expected_results = [
            r'Epoch:   0 (.*) | Train:   1.90 % | Validation:   2.50 % | Test:   3.00 %',
            r'Epoch:   1 (.*) | Train:   2.20 % | Validation:   1.50 % | Test:   2.20 %',
            r'Epoch:   2 (.*) | Train:   3.20 % | Validation:   4.50 % | Test:   5.00 %',
        ]
    elif device == 'cpu':
        expected_results = [
            r'Epoch:   0 (.*) | Train:   2.00 % | Validation:   2.00 % | Test:   2.00 %',
            r'Epoch:   1 (.*) | Train:   2.30 % | Validation:   3.70 % | Test:   2.50 %',
            r'Epoch:   2 (.*) | Train:   4.80 % | Validation:   4.50 % | Test:   4.50 %',
        ]

    os.chdir('examples')
    cmd = ['./esc_convnet_paper.py', '-D', device]
    popen = subprocess.Popen(cmd, stdout=PIPE, stderr=STDOUT, universal_newlines=True)

    verified_epochs = 0

    for line in iter(popen.stdout.readline, ""):
        if line.startswith('Epoch: '):
            print('.', end='', flush=True)
            assert re.search(expected_results[verified_epochs], line) is not None
            verified_epochs += 1

        if (device == 'cpu' and verified_epochs > 1) or verified_epochs > 2:
            break


if __name__ == '__main__':
    pytest.main([__file__])
