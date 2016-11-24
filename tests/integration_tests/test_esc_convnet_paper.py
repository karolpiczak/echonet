#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import subprocess

from subprocess import STDOUT

import pytest


def test_results_reproducibility(capsys, device):
    os.chdir('examples')
    try:
        cmd = ['./esc_convnet_paper.py', '-D', device]
        out = subprocess.check_output(cmd, stderr=STDOUT, timeout=900)
    except subprocess.TimeoutExpired as e:
        out = e.output
    out = out.decode('utf-8')

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

    for line in expected_results:
        assert re.search(line, out) is not None


if __name__ == '__main__':
    pytest.main([__file__])
