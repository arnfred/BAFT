## README - DAFT: Direction Agnostic Feature Transform

DAFT is a local image feature descriptor. At least that's the idea. We aren't
quite there yet.

Version: 0.0.1
Date: 2014-11-27

You can get the latest version of the code from github:
`https://github.com/arnfred/DAFT`

## What is this file?

This file explains how to make use of the OpenCV KAZE and A-KAZE features code in
an image matching application

## Library Dependencies

The code is based on the OpenCV library using the C++ interface. You will need
to download and install the master branch version from OpenCV github repository.
`https://github.com/Itseez/opencv`

## Getting Started

Compiling:

1. `$ mkdir build`
2. `$ cd build>`
3. `$ cmake ..`
4. `$ make`

You should see the executable `test_daft` after compiling.
