# 02460 Project
Advanced Machine Learning project

By Jesper Hybel, Søren Jensen, Lorenzo Belgrano and Mirza Hasanbasic

Table of Contents
=================

# About this project

## How to run the code

For the DTD part one should download the DTD dataset, that can be found by clicking [here](https://www.robots.ox.ac.uk/~vgg/data/dtd/index.html) and the VGG16 weights can be found by clicking [here](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3).

The folder FV-CNN should be accessed to run the Fisher Vector CNN part. You should place the pictures of the DTD in the DTD folder. Futhermore the weights should be in the FV-CNN folder too, note that the .npz format should be placed here and not the .h5 format.

The code is written in python, to be able to run the `fvcnn_full.py` one should have these python packages install:


`Tensorflow`

`Matlab-python`

`scikit-learn`

`numpy`

`scipy`

`pickle`

`sklearn`


From approx line 410-415 you can state to load all pics then set `pics_to_load = None` and the variable `to_do_folds` will set how many folds to run.

### How to install Matlab-python

To install the matlab-python engine, please visit this link, by clicking [here](https://se.mathworks.com/help/matlab/matlab-engine-for-python.html)

### MCG

Install this package for the MCG: https://github.com/jponttuset/mcg


