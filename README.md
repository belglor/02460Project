# BAG OF WORDS MODEL FOR TEXTURE CLASSIFICATION

Link to the final article: https://github.com/belglor/02460Project/blob/master/BAG%20OF%20WORDS%20MODEL%20FOR%20TEXTURE%20CLASSIFICATION.pdf

**Abstract** In this project we present and illustrate a method of picture segmentation based on texture recognition and show that the texture recognition model is transferable from one dataset to another. The method for picture segmentation combines a region proposal algorithm with a texture recognition method based on the convolutional layers of VGG-16 in combination with ﬁscher vectors. We test the texture recognition method on the uncluttered textures using Describable Textures Dataset (DTD) and transfer it on cluttered textures using the OpenSurfaces (OS) dataset allowing only partial retraining. We achieve a classiﬁcation accuracy of 64.0% on the DTD dataset and 56.48% on the OS dataset. 

**Index Terms-** Convolutional neural networks , Fisher Vectors, VGG-16, Texture Classiﬁcation, Benchmark

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

In the `test_code` one could find code examples to run the different parts of the algorithm: 
 - [The VGG-16 netowrk](https://github.com/belglor/02460Project/tree/master/test_code/VGG16%20test)
 - [Fisher vectors](https://github.com/belglor/02460Project/tree/master/test_code/FisherVector%20test)
 - [GMM clustering](https://github.com/belglor/02460Project/tree/master/test_code/gaussianmixturemodels)
 

### How to install Matlab-python

To install the matlab-python engine, please visit this link, by clicking [here](https://se.mathworks.com/help/matlab/matlab-engine-for-python.html)

### MCG

Install this package for the MCG: https://github.com/jponttuset/mcg





