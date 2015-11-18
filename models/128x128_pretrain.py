import os
import sys
import re
p = re.compile("/home/jilee/projects/*")
for path in sys.path:
    if p.match(path):
        sys.path.remove(path)
import time
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda import dnn

import lasagne
from lasagne.layers import DenseLayer, InputLayer, FeaturePoolLayer, DropoutLayer
from lasagne.updates import nesterov_momentum
from lasagne.objectives import squared_error, aggregate
from lasagne import init, layers
from lasagne.nonlinearities import rectify, leaky_rectify

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import objective

import iterator
import data
from layers import RMSPoolLayer

try:
    import lasagne.layers.dnn
    Conv2DLayer = lasagne.layers.dnn.Conv2DDNNLayer
    MaxPool2DLayer = lasagne.layers.dnn.MaxPool2DDNNLayer
    Pool2DLayer = lasagne.layers.dnn.Pool2DDNNLayer
    print("using CUDNN backend")
except ImportError:
    print("failed to load CUDNN backend")
    Conv2DLayer = lasagne.layers.conv.Conv2DLayer
    MaxPool2DLayer = lasagne.layers.pool.Pool2DLayer
    MaxPool2DLayer = MaxPool2DLayer

aug_params = {
    'zoom_range': (1 / 1.15, 1.15),
    'rotation_range': (0, 360),
    'shear_range': (0, 0),
    'translation_range': (-40, 40),
    'do_flip': True,
    'allow_stretch': True,
}


layers0 = [
    ('input', InputLayer),
    ('conv1', Conv2DLayer),
    ('conv2', Conv2DLayer),
    ('pool1', MaxPool2DLayer),
    ('conv3', Conv2DLayer),
    ('conv4', Conv2DLayer),
    ('conv5', Conv2DLayer),
    ('pool2', MaxPool2DLayer),
    ('conv6', Conv2DLayer),
    ('conv7', Conv2DLayer),
    ('conv8', Conv2DLayer),
    ('pool3', RMSPoolLayer),
    ('dropout1', DropoutLayer),
    ('fc1', DenseLayer),
    ('pool4', FeaturePoolLayer),
    ('dropout2', DropoutLayer),
    ('fc2', DenseLayer),
    ('pool5', FeaturePoolLayer),
    ('output', DenseLayer)
]

def build_model():
    net = NeuralNet(
        layers=layers0,
        input_shape=(None, 3, 112, 112),
        conv1_num_filters=32, conv1_filter_size=(5, 5), conv1_stride=(2, 2), conv1_pad='same', conv1_W=init.Orthogonal('relu'),
        conv2_num_filters=32, conv2_filter_size=(3, 3), conv2_stride=(1, 1), conv2_pad='same', conv2_W=init.Orthogonal('relu'),
        pool1_pool_size=(3, 3), pool1_stride=(2, 2),
        conv3_num_filters=64, conv3_filter_size=(5, 5), conv3_stride=(2, 2), conv3_pad='same', conv3_W=init.Orthogonal('relu'),
        conv4_num_filters=64, conv4_filter_size=(3, 3), conv4_stride=(1, 1), conv4_pad='same', conv4_W=init.Orthogonal('relu'),
        conv5_num_filters=64, conv5_filter_size=(3, 3), conv5_stride=(1, 1), conv5_pad='same', conv5_W=init.Orthogonal('relu'),
        pool2_pool_size=(3, 3), pool2_stride=(2, 2),
        conv6_num_filters=128, conv6_filter_size=(3, 3), conv6_stride=(1, 1), conv6_pad='same', conv6_W=init.Orthogonal('relu'),
        conv7_num_filters=128, conv7_filter_size=(3, 3), conv7_stride=(1, 1), conv7_pad='same', conv7_W=init.Orthogonal('relu'),
        conv8_num_filters=128, conv8_filter_size=(3, 3), conv8_stride=(1, 1), conv8_pad='same', conv8_W=init.Orthogonal('relu'),
        pool3_pool_size=(3, 3), pool3_stride=(3, 3),
        dropout1_p=0.5,
        fc1_num_units=1024, fc1_nonlinearity=rectify, fc1_W=init.Orthogonal('relu'),
        pool4_pool_size=2,
        dropout2_p=0.5,
        fc2_num_units=1024, fc2_nonlinearity=rectify, fc2_W=init.Orthogonal('relu'),
        pool5_pool_size=2,
        output_num_units=1,

        update=nesterov_momentum,
        update_learning_rate=0.0001,

        regression=True,
        objective=objective,
        objective_loss_function=squared_error,
        objective_aggregate=aggregate,
        objective_l2=0.005,

        batch_iterator_train =
           iterator.BatchIteratorAugmented(128,
                                           transform=data.perturb_and_augment,
                                           aug_params=aug_params,
                                           sigma=0.5),
        batch_iterator_test =
           iterator.BatchIteratorAugmented(128,
                                           transform=data.perturb_and_augment),

        max_epochs=30000,
        verbose=2
    )
    return net












