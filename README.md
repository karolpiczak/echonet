# EchoNet

[![Travis](https://img.shields.io/travis/karoldvl/echonet.svg)](https://travis-ci.org/karoldvl/echonet)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/karoldvl/echonet/master/LICENSE)
[![Waffle.io](https://img.shields.io/waffle/label/karoldvl/echonet/to-do.svg)](http://waffle.io/karoldvl/echonet)

Porting some of my old convnet code into Keras - a work in progress.

It's still a very rough skeleton, so use at your own risk, but due to numerous inquiries I decided to publish it sooner rather than later. I will try to implement the missing bits soon.

Current features:
- `examples/esc_convnet_paper.py` reimplements the [Environmental Sound Classification with Convolutional Neural Networks paper](https://github.com/karoldvl/paper-2015-esc-convnet)
- `echonet/datasets/esc_original.py` provides a loading wrapper around the ESC-50 dataset

For the time being some important requirements:
- Python 3.5
- ESC-50 WAV version: https://github.com/karoldvl/ESC-50/archive/wav-files.zip
- Install dependencies with ```sudo pip install -r requirements.txt```
