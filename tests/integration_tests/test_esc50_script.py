#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test reproducibility of the original implementation of the ESC-ConvNet paper."""

import os
import re
import subprocess

from subprocess import PIPE, STDOUT

import pytest


@pytest.mark.timeout(2400)
def test_results_reproducibility(device):
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

    os.chdir('experiments')
    cmd = ['./esc50.py', '-D', device, 'test-esc50']
    popen = subprocess.Popen(cmd, stdout=PIPE, stderr=STDOUT, universal_newlines=True)

    verified_epochs = 0

    for line in iter(popen.stdout.readline, ""):
        if line.startswith('Epoch: '):
            assert re.search(expected_results[verified_epochs], line) is not None
            verified_epochs += 1

        if (device == 'cpu' and verified_epochs > 1) or verified_epochs > 2:
            break


if __name__ == '__main__':
    pytest.main([__file__])
