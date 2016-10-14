# -*- coding: utf-8 -*-
__author__ = 'Rainer Arencibia'
"""
MIT License

Copyright (c) 2016 Rainer Arencibia

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from keras.layers import noise
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential


# from DeepLearning.cnn.network.dropconnect import DropConnect


class LeNet:
    def __init__(self):
        pass

    @staticmethod
    def build(width, height, depth, classes, weights_path):
        """
        :param width: the width of the input images
        :param height: the height of the input images
        :param depth:  the depth of the input images
        :param classes: the numbers of labels
        :param weights_path: URL of an already trained model.
        :return: an empty model.
        """

        # initialize the model..
        model = Sequential()

        # first set of CONV => RELU => POOL
        model.add(Convolution2D(32, 5, 5, init='lecun_uniform', border_mode="same", bias=True,
                                input_shape=(depth, height, width)))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL => Dropout
        model.add(Convolution2D(64, 5, 5, init='lecun_uniform', border_mode="same", bias=True))
        model.add(BatchNormalization())
        model.add(LeakyReLU())  # model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        noise.GaussianDropout(0.25)

        # set of FC => RELU layers
        model.add(Flatten())  # convert convolutional filters to flatt so they can be feed to fully connected layers

        # first fully connected layer
        model.add(Dense(400, init='lecun_uniform', bias=True))  # init='glorot_uniform'
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        noise.GaussianDropout(0.25)  # model.add(Dropout(0.25))

        # second fully connected layer.. softmax classifier
        model.add(Dense(classes, init='lecun_uniform', bias=True))
        model.add(BatchNormalization())
        model.add(Activation("softmax"))

        # if a weights path is supplied (indicating that the model was pre-trained), then load the weights.
        if weights_path is not None:
            model.load_weights(weights_path)

        # return the constructed network architecture
        return model


"""
--weights
/home/rainer85ah/PycharmProjects/PythonProjects/Output/lenet_weights_9936.hdf5
/usr/bin/python2.7 /home/rainer85ah/PycharmProjects/PythonProjects/DeepLearning/lenet_mnist.py --save-model 1 --weights
/home/rainer85ah/PycharmProjects/PythonProjects/Output/lenet_weights_9936.hdf5
Using TensorFlow backend.
[INFO] downloading MNIST...
[INFO] Summary of the model...
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
convolution2d_1 (Convolution2D)  (None, 32, 28, 28)    832         convolution2d_input_1[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 32, 28, 28)    0           convolution2d_1[0][0]
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 32, 14, 14)    0           activation_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 64, 14, 14)    51264       maxpooling2d_1[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 64, 14, 14)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 64, 7, 7)      0           activation_2[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 64, 7, 7)      0           maxpooling2d_2[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 3136)          0           dropout_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 400)           1254800     flatten_1[0][0]
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 400)           0           dense_1[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 400)           0           activation_3[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 10)            4010        dropout_2[0][0]
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 10)            0           dense_2[0][0]
====================================================================================================
Total params: 1310906
____________________________________________________________________________________________________
[INFO] compiling model...
[INFO] training...
Train on 50575 samples, validate on 8925 samples
Epoch 1/100
50575/50575 [==============================] - 445s - loss: 0.2176 - acc: 0.9314 - val_loss: 0.0417 - val_acc: 0.9861
Epoch 2/100
50575/50575 [==============================] - 490s - loss: 0.0540 - acc: 0.9830 - val_loss: 0.1167 - val_acc: 0.9669
Epoch 3/100
50575/50575 [==============================] - 492s - loss: 0.0453 - acc: 0.9861 - val_loss: 0.0325 - val_acc: 0.9900
Epoch 4/100
50575/50575 [==============================] - 493s - loss: 0.0343 - acc: 0.9895 - val_loss: 0.0317 - val_acc: 0.9915
Epoch 5/100
50575/50575 [==============================] - 491s - loss: 0.0292 - acc: 0.9903 - val_loss: 0.0247 - val_acc: 0.9918
Epoch 6/100
50575/50575 [==============================] - 496s - loss: 0.0259 - acc: 0.9920 - val_loss: 0.0244 - val_acc: 0.9929
Epoch 7/100
50575/50575 [==============================] - 492s - loss: 0.0235 - acc: 0.9922 - val_loss: 0.0301 - val_acc: 0.9907
Epoch 8/100
50575/50575 [==============================] - 491s - loss: 0.0177 - acc: 0.9945 - val_loss: 0.0255 - val_acc: 0.9920
Epoch 9/100
50575/50575 [==============================] - 492s - loss: 0.0182 - acc: 0.9944 - val_loss: 0.0262 - val_acc: 0.9922
Epoch 10/100
50575/50575 [==============================] - 490s - loss: 0.0147 - acc: 0.9951 - val_loss: 0.0331 - val_acc: 0.9919
Epoch 11/100
50575/50575 [==============================] - 492s - loss: 0.0145 - acc: 0.9954 - val_loss: 0.0251 - val_acc: 0.9937
Epoch 12/100
50575/50575 [==============================] - 515s - loss: 0.0130 - acc: 0.9959 - val_loss: 0.0247 - val_acc: 0.9938
Epoch 13/100
50575/50575 [==============================] - 552s - loss: 0.0154 - acc: 0.9952 - val_loss: 0.0297 - val_acc: 0.9919
Epoch 14/100
50575/50575 [==============================] - 472s - loss: 0.0137 - acc: 0.9957 - val_loss: 0.0282 - val_acc: 0.9927
Epoch 15/100
50575/50575 [==============================] - 450s - loss: 0.0114 - acc: 0.9962 - val_loss: 0.0251 - val_acc: 0.9929
Epoch 16/100
50575/50575 [==============================] - 470s - loss: 0.0126 - acc: 0.9961 - val_loss: 0.0344 - val_acc: 0.9924
Epoch 17/100
50575/50575 [==============================] - 462s - loss: 0.0107 - acc: 0.9967 - val_loss: 0.0272 - val_acc: 0.9926
Epoch 18/100
50575/50575 [==============================] - 460s - loss: 0.0103 - acc: 0.9967 - val_loss: 0.0326 - val_acc: 0.9920
Epoch 19/100
50575/50575 [==============================] - 461s - loss: 0.0090 - acc: 0.9970 - val_loss: 0.0338 - val_acc: 0.9917
Epoch 20/100
50575/50575 [==============================] - 479s - loss: 0.0094 - acc: 0.9973 - val_loss: 0.0294 - val_acc: 0.9935
Epoch 21/100
50575/50575 [==============================] - 467s - loss: 0.0088 - acc: 0.9970 - val_loss: 0.0291 - val_acc: 0.9931
Epoch 22/100
50575/50575 [==============================] - 468s - loss: 0.0082 - acc: 0.9973 - val_loss: 0.0249 - val_acc: 0.9929
Epoch 23/100
50575/50575 [==============================] - 432s - loss: 0.0079 - acc: 0.9974 - val_loss: 0.0303 - val_acc: 0.9926
Epoch 24/100
50575/50575 [==============================] - 481s - loss: 0.0079 - acc: 0.9975 - val_loss: 0.0326 - val_acc: 0.9925
Epoch 25/100
50575/50575 [==============================] - 476s - loss: 0.0073 - acc: 0.9979 - val_loss: 0.0346 - val_acc: 0.9915
Epoch 26/100
50575/50575 [==============================] - 492s - loss: 0.0058 - acc: 0.9979 - val_loss: 0.0312 - val_acc: 0.9941
Epoch 27/100
50575/50575 [==============================] - 444s - loss: 0.0079 - acc: 0.9975 - val_loss: 0.0373 - val_acc: 0.9924
Epoch 28/100
50575/50575 [==============================] - 431s - loss: 0.0078 - acc: 0.9976 - val_loss: 0.0316 - val_acc: 0.9928
Epoch 29/100
50575/50575 [==============================] - 444s - loss: 0.0052 - acc: 0.9985 - val_loss: 0.0302 - val_acc: 0.9933
Epoch 30/100
50575/50575 [==============================] - 428s - loss: 0.0052 - acc: 0.9985 - val_loss: 0.0364 - val_acc: 0.9934
Epoch 31/100
50575/50575 [==============================] - 445s - loss: 0.0060 - acc: 0.9979 - val_loss: 0.0322 - val_acc: 0.9943
Epoch 32/100
50575/50575 [==============================] - 428s - loss: 0.0086 - acc: 0.9976 - val_loss: 0.0307 - val_acc: 0.9934
Epoch 33/100
50575/50575 [==============================] - 432s - loss: 0.0080 - acc: 0.9977 - val_loss: 0.0306 - val_acc: 0.9942
Epoch 34/100
50575/50575 [==============================] - 431s - loss: 0.0066 - acc: 0.9979 - val_loss: 0.0299 - val_acc: 0.9935
Epoch 35/100
50575/50575 [==============================] - 431s - loss: 0.0077 - acc: 0.9976 - val_loss: 0.0285 - val_acc: 0.9947
Epoch 36/100
50575/50575 [==============================] - 431s - loss: 0.0062 - acc: 0.9981 - val_loss: 0.0350 - val_acc: 0.9935
Epoch 37/100
50575/50575 [==============================] - 431s - loss: 0.0060 - acc: 0.9982 - val_loss: 0.0320 - val_acc: 0.9938
Epoch 38/100
50575/50575 [==============================] - 431s - loss: 0.0066 - acc: 0.9981 - val_loss: 0.0312 - val_acc: 0.9934
Epoch 39/100
50575/50575 [==============================] - 431s - loss: 0.0053 - acc: 0.9985 - val_loss: 0.0367 - val_acc: 0.9934
Epoch 40/100
50575/50575 [==============================] - 431s - loss: 0.0059 - acc: 0.9986 - val_loss: 0.0345 - val_acc: 0.9941
Epoch 41/100
50575/50575 [==============================] - 431s - loss: 0.0049 - acc: 0.9986 - val_loss: 0.0311 - val_acc: 0.9948
Epoch 42/100
50575/50575 [==============================] - 431s - loss: 0.0047 - acc: 0.9987 - val_loss: 0.0390 - val_acc: 0.9938
Epoch 43/100
50575/50575 [==============================] - 431s - loss: 0.0067 - acc: 0.9981 - val_loss: 0.0419 - val_acc: 0.9932
Epoch 44/100
50575/50575 [==============================] - 431s - loss: 0.0065 - acc: 0.9982 - val_loss: 0.0329 - val_acc: 0.9941
Epoch 45/100
50575/50575 [==============================] - 431s - loss: 0.0054 - acc: 0.9983 - val_loss: 0.0321 - val_acc: 0.9932
Epoch 46/100
50575/50575 [==============================] - 431s - loss: 0.0068 - acc: 0.9981 - val_loss: 0.0282 - val_acc: 0.9944
Epoch 47/100
50575/50575 [==============================] - 431s - loss: 0.0054 - acc: 0.9987 - val_loss: 0.0339 - val_acc: 0.9939
Epoch 48/100
50575/50575 [==============================] - 433s - loss: 0.0062 - acc: 0.9982 - val_loss: 0.0524 - val_acc: 0.9909
Epoch 49/100
50575/50575 [==============================] - 435s - loss: 0.0063 - acc: 0.9983 - val_loss: 0.0395 - val_acc: 0.9941
Epoch 50/100
50575/50575 [==============================] - 443s - loss: 0.0059 - acc: 0.9983 - val_loss: 0.0315 - val_acc: 0.9942
Epoch 51/100
50575/50575 [==============================] - 431s - loss: 0.0052 - acc: 0.9985 - val_loss: 0.0431 - val_acc: 0.9929
Epoch 52/100
50575/50575 [==============================] - 432s - loss: 0.0054 - acc: 0.9984 - val_loss: 0.0400 - val_acc: 0.9936
Epoch 53/100
50575/50575 [==============================] - 431s - loss: 0.0076 - acc: 0.9979 - val_loss: 0.0351 - val_acc: 0.9941
Epoch 54/100
50575/50575 [==============================] - 431s - loss: 0.0059 - acc: 0.9984 - val_loss: 0.0315 - val_acc: 0.9938
Epoch 55/100
50575/50575 [==============================] - 431s - loss: 0.0049 - acc: 0.9985 - val_loss: 0.0385 - val_acc: 0.9935
Epoch 56/100
50575/50575 [==============================] - 431s - loss: 0.0059 - acc: 0.9987 - val_loss: 0.0413 - val_acc: 0.9936
Epoch 57/100
50575/50575 [==============================] - 431s - loss: 0.0084 - acc: 0.9977 - val_loss: 0.0525 - val_acc: 0.9916
Epoch 58/100
50575/50575 [==============================] - 431s - loss: 0.0058 - acc: 0.9985 - val_loss: 0.0389 - val_acc: 0.9931
Epoch 59/100
50575/50575 [==============================] - 431s - loss: 0.0067 - acc: 0.9982 - val_loss: 0.0443 - val_acc: 0.9934
Epoch 60/100
50575/50575 [==============================] - 432s - loss: 0.0061 - acc: 0.9984 - val_loss: 0.0377 - val_acc: 0.9932
Epoch 61/100
50575/50575 [==============================] - 433s - loss: 0.0034 - acc: 0.9990 - val_loss: 0.0418 - val_acc: 0.9931
Epoch 62/100
50575/50575 [==============================] - 448s - loss: 0.0045 - acc: 0.9987 - val_loss: 0.0385 - val_acc: 0.9937
Epoch 63/100
50575/50575 [==============================] - 443s - loss: 0.0038 - acc: 0.9989 - val_loss: 0.0482 - val_acc: 0.9942
Epoch 64/100
50575/50575 [==============================] - 428s - loss: 0.0087 - acc: 0.9978 - val_loss: 0.0327 - val_acc: 0.9945
Epoch 65/100
50575/50575 [==============================] - 432s - loss: 0.0068 - acc: 0.9983 - val_loss: 0.0409 - val_acc: 0.9936
Epoch 66/100
50575/50575 [==============================] - 428s - loss: 0.0049 - acc: 0.9987 - val_loss: 0.0365 - val_acc: 0.9936
Epoch 67/100
50575/50575 [==============================] - 428s - loss: 0.0046 - acc: 0.9990 - val_loss: 0.0321 - val_acc: 0.9937
Epoch 68/100
50575/50575 [==============================] - 428s - loss: 0.0033 - acc: 0.9991 - val_loss: 0.0416 - val_acc: 0.9941
Epoch 69/100
50575/50575 [==============================] - 434s - loss: 0.0030 - acc: 0.9991 - val_loss: 0.0392 - val_acc: 0.9931
Epoch 70/100
50575/50575 [==============================] - 440s - loss: 0.0063 - acc: 0.9983 - val_loss: 0.0412 - val_acc: 0.9937
Epoch 71/100
50575/50575 [==============================] - 433s - loss: 0.0069 - acc: 0.9981 - val_loss: 0.0429 - val_acc: 0.9932
Epoch 72/100
50575/50575 [==============================] - 431s - loss: 0.0052 - acc: 0.9986 - val_loss: 0.0374 - val_acc: 0.9939
Epoch 73/100
50575/50575 [==============================] - 450s - loss: 0.0047 - acc: 0.9987 - val_loss: 0.0426 - val_acc: 0.9938
Epoch 74/100
50575/50575 [==============================] - 469s - loss: 0.0074 - acc: 0.9980 - val_loss: 0.0496 - val_acc: 0.9920
Epoch 75/100
50575/50575 [==============================] - 473s - loss: 0.0051 - acc: 0.9986 - val_loss: 0.0468 - val_acc: 0.9929
Epoch 76/100
50575/50575 [==============================] - 499s - loss: 0.0079 - acc: 0.9981 - val_loss: 0.0384 - val_acc: 0.9931
Epoch 77/100
50575/50575 [==============================] - 454s - loss: 0.0054 - acc: 0.9988 - val_loss: 0.0409 - val_acc: 0.9936
Epoch 78/100
50575/50575 [==============================] - 433s - loss: 0.0043 - acc: 0.9988 - val_loss: 0.0404 - val_acc: 0.9938
Epoch 79/100
50575/50575 [==============================] - 431s - loss: 0.0076 - acc: 0.9983 - val_loss: 0.0463 - val_acc: 0.9923
Epoch 80/100
50575/50575 [==============================] - 426s - loss: 0.0085 - acc: 0.9978 - val_loss: 0.0528 - val_acc: 0.9923
Epoch 81/100
50575/50575 [==============================] - 427s - loss: 0.0076 - acc: 0.9980 - val_loss: 0.0507 - val_acc: 0.9938
Epoch 82/100
50575/50575 [==============================] - 426s - loss: 0.0122 - acc: 0.9972 - val_loss: 0.0437 - val_acc: 0.9927
Epoch 83/100
50575/50575 [==============================] - 426s - loss: 0.0108 - acc: 0.9974 - val_loss: 0.0458 - val_acc: 0.9931
Epoch 84/100
50575/50575 [==============================] - 426s - loss: 0.0079 - acc: 0.9980 - val_loss: 0.0442 - val_acc: 0.9925
Epoch 85/100
50575/50575 [==============================] - 427s - loss: 0.0073 - acc: 0.9981 - val_loss: 0.0418 - val_acc: 0.9932
Epoch 86/100
50575/50575 [==============================] - 427s - loss: 0.0085 - acc: 0.9979 - val_loss: 0.0540 - val_acc: 0.9929
Epoch 87/100
50575/50575 [==============================] - 426s - loss: 0.0101 - acc: 0.9976 - val_loss: 0.0576 - val_acc: 0.9933
Epoch 88/100
50575/50575 [==============================] - 426s - loss: 0.0089 - acc: 0.9978 - val_loss: 0.0447 - val_acc: 0.9937
Epoch 89/100
50575/50575 [==============================] - 426s - loss: 0.0091 - acc: 0.9980 - val_loss: 0.0424 - val_acc: 0.9934
Epoch 90/100
50575/50575 [==============================] - 426s - loss: 0.0098 - acc: 0.9978 - val_loss: 0.0638 - val_acc: 0.9914
Epoch 91/100
50575/50575 [==============================] - 426s - loss: 0.0121 - acc: 0.9972 - val_loss: 0.0561 - val_acc: 0.9920
Epoch 92/100
50575/50575 [==============================] - 427s - loss: 0.0138 - acc: 0.9973 - val_loss: 0.0589 - val_acc: 0.9924
Epoch 93/100
50575/50575 [==============================] - 446s - loss: 0.0126 - acc: 0.9974 - val_loss: 0.0519 - val_acc: 0.9918
Epoch 94/100
50575/50575 [==============================] - 426s - loss: 0.0144 - acc: 0.9970 - val_loss: 0.0539 - val_acc: 0.9924
Epoch 95/100
50575/50575 [==============================] - 426s - loss: 0.0114 - acc: 0.9977 - val_loss: 0.0498 - val_acc: 0.9918
Epoch 96/100
50575/50575 [==============================] - 426s - loss: 0.0067 - acc: 0.9982 - val_loss: 0.0567 - val_acc: 0.9924
Epoch 97/100
50575/50575 [==============================] - 427s - loss: 0.0114 - acc: 0.9974 - val_loss: 0.0535 - val_acc: 0.9919
Epoch 98/100
50575/50575 [==============================] - 426s - loss: 0.0174 - acc: 0.9967 - val_loss: 0.0524 - val_acc: 0.9927
Epoch 99/100
50575/50575 [==============================] - 427s - loss: 0.0158 - acc: 0.9969 - val_loss: 0.0610 - val_acc: 0.9917
Epoch 100/100
50575/50575 [==============================] - 426s - loss: 0.0114 - acc: 0.9977 - val_loss: 0.0520 - val_acc: 0.9936
[INFO] evaluating...
10500/10500 [==============================] - 34s
[INFO] accuracy: 99.36%
[INFO] dumping weights to file...

97.50%, 98.45%, 98.49%, 99.05%, 98,98%, 99.22%(3M), 99.28%(1.3M), 98.93%(455k), 98.99%(1.3M), 99.36%, 99.23%,
99.21% (3.7M), 99.30%,
"""
