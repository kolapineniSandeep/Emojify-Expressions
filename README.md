# Emojify-Expressions
Emojify-Expressions Face2013 is a machine learning project that uses facial expression recognition to map a human's emotions to emojis. The project is based on the Face2013 dataset and uses various image processing and machine learning algorithms to predict the most suitable emoji for a given expression.

## Getting Started

To get started with Emojify-Expressions Face2013, you'll need to have Python 3.x and several Python packages installed on your system. We recommend using a virtual environment(Anaconda) to manage your dependencies and run GUI. You can set up a virtual environment and install the required packages by running:

## Demo

https://user-images.githubusercontent.com/98118151/232617476-5c522cfd-51ca-4d45-8c80-59d690e59718.mp4

## CNN Model Training for Image Classification

This repository contains code for training a Convolutional Neural Network (CNN) for image classification. The code is designed to work with the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes.

### Requirements

To run the code in this repository, you'll need the following dependencies:

Python 3.x
TensorFlow 2.x
NumPy
Matplotlib


### Model Architecture

The CNN used in this repository has the following architecture:

```
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 30, 30, 32)        896
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 13, 13, 64)        18496
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 6, 6, 64)          0
_________________________________________________________________
flatten (Flatten)            (None, 2304)              0
_________________________________________________________________
dense (Dense)                (None, 128)               295040
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290
=================================================================
Total params: 315,722
Trainable params: 315,722
Non-trainable params: 0
```

### Results

After training the CNN for 50 epochs, we achieved an accuracy of 80% on the validation set. We also observed that the model had a tendency to overfit the training data, which suggests that additional regularization techniques such as dropout or data augmentation could be beneficial.
