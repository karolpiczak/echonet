#!/bin/bash

python -m pytest tests/ --ignore=tests/integration_tests
python -m pytest --pep8 -m pep8 -n0
python -m pytest tests/integration_tests --GPU
