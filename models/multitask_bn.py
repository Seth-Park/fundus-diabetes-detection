import os

import numpy as np
import pandas as p
import theano.tensor as T
import lasagne as nn
from lasagne.layers import dnn
from lasagne.nonlinearities import LeakyRectify, softmax, identity, sigmoid
from batch_norm import *

Conv2DLayer = dnn.Conv2DDNNLayer
MaxPool2DLayer = dnn.MaxPool2DDNNLayer
DenseLayer = nn.layers.DenseLayer

input_size = 512
input_height, input_width = (input_size, input_size)
output_dim = 5
num_channels = 3

leakiness = 0.5

def build_model(input_var):

    layers = []

    input_layer = nn.layers.InputLayer(
            shape=(None, num_channels, input_width, input_height),
            input_var=input_var,
            name='inputs'
    )
    layers.append(input_layer)

    conv_1 = batch_norm(Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(7, 7), stride=(2, 2),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True))
    layers.append(conv_1)

    pool_1 = batch_norm(Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(3, 3), stride=(2, 2),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True))
                         #MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(pool_1)

    conv_2 = batch_norm(Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True))
    layers.append(conv_2)

    conv_3 = batch_norm(Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True))
    layers.append(conv_3)

    #pool_2 = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    pool_2 = batch_norm(Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(3, 3), stride=(2, 2),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True))
    layers.append(pool_2)

    conv_4 = batch_norm(Conv2DLayer(layers[-1],
                         num_filters=64, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True))
    layers.append(conv_4)

    conv_5 = batch_norm(Conv2DLayer(layers[-1],
                         num_filters=64, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True))
    layers.append(conv_5)

    #pool_3 = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    pool_3 = batch_norm(Conv2DLayer(layers[-1],
                         num_filters=64, filter_size=(3, 3), stride=(2, 2),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True))
    layers.append(pool_3)

    conv_6 = batch_norm(Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True))
    layers.append(conv_6)

    conv_7 = batch_norm(Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True))
    layers.append(conv_7)

    conv_8 = batch_norm(Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True))
    layers.append(conv_8)

    conv_9 = batch_norm(Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True))
    layers.append(conv_9)

    #pool_4 = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    pool_4 = batch_norm(Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(2, 2),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True))
    layers.append(pool_4)

    conv_10 = batch_norm(Conv2DLayer(layers[-1],
                          num_filters=256, filter_size=(3, 3), stride=(1, 1),
                          pad='same',
                          nonlinearity=LeakyRectify(leakiness),
                          W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                          untie_biases=True))
    layers.append(conv_10)

    conv_11 = batch_norm(Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True))
    layers.append(conv_11)

    conv_12 = batch_norm(Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True))
    layers.append(conv_12)

    conv_13 = batch_norm(Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True))
    layers.append(conv_13)

    #pool_5 = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    pool_5 = batch_norm(Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(2, 2),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True))
    layers.append(pool_5)

    drop_1 = nn.layers.DropoutLayer(layers[-1], p=0.5)
    layers.append(drop_1)

    fc_1 = DenseLayer(layers[-1],
                      num_units=2048,
                      nonlinearity=LeakyRectify(leakiness),
                      W=nn.init.Orthogonal(1.0),
                      b=nn.init.Constant(0.1))
    layers.append(fc_1)

    pool_6 = nn.layers.FeaturePoolLayer(layers[-1],
                                        pool_size=2,
                                        pool_function=T.max)
    layers.append(pool_6)

    drop_2 = nn.layers.DropoutLayer(layers[-1], p=0.5)
    layers.append(drop_2)

    fc_2 = DenseLayer(layers[-1],
                      num_units=1024,
                      nonlinearity=LeakyRectify(leakiness),
                      W=nn.init.Orthogonal(1.0),
                      b=nn.init.Constant(0.1))
    layers.append(fc_2)

    pool_7 = nn.layers.FeaturePoolLayer(layers[-1],
                                        pool_size=2,
                                        pool_function=T.max)
    layers.append(pool_7)

    drop_3 = nn.layers.DropoutLayer(layers[-1], p=0.5)
    layers.append(drop_3)

    softmax_layer = DenseLayer(drop_3,
                      num_units=output_dim,
                      nonlinearity=softmax)
    layers.append(softmax_layer)

    regression_layer = DenseLayer(drop_3,
                        num_units=1,
                        nonlinearity=identity)
    layers.append(regression_layer)

    output = nn.layers.merge.ConcatLayer([softmax_layer, regression_layer])
    layers.append(output)

    return input_layer, output

