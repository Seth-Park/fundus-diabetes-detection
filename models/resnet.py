import numpy as np
import theano
import theano.tensor as T
import lasagne as nn

from lasagne.nonlinearities import rectify, softmax
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import Pool2DDNNLayer as PoolLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPoolLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import PadLayer
from lasagne.layers import NonlinearityLayer

from batch_norm import *

##### Network Config #####
input_size = 256
input_height, input_width = (input_size, input_size)
output_dim = 5
num_channels = 3
batch_size = 16
num_blocks = [3, 4, 6, 3]

##### Build ResNet #####
def build_model(input_var):

    # Three layer residual block
    def residual_block3(l, base_dim, increase_dim=False, projection=False):
        if increase_dim:
            layer_1 = batch_norm(ConvLayer(l,
                                           num_filters=base_dim,
                                           filter_size=(1, 1),
                                           stride=(2, 2),
                                           nonlinearity=None,
                                           pad='same',
                                           W=nn.init.HeNormal(gain='relu')))
        else:
            layer_1 = batch_norm(ConvLayer(l,
                                           num_filters=base_dim,
                                           filter_size=(1, 1),
                                           stride=(1, 1),
                                           nonlinearity=rectify,
                                           pad='same',
                                           W=nn.init.HeNormal(gain='relu')))
        layer_2 = batch_norm(ConvLayer(layer_1,
                                       num_filters=base_dim,
                                       filter_size=(3, 3),
                                       stride=(1, 1),
                                       nonlinearity=rectify,
                                       pad='same',
                                       W=nn.init.HeNormal(gain='relu')))
        layer_3 = batch_norm(ConvLayer(layer_2,
                                       num_filters=4*base_dim,
                                       filter_size=(1, 1),
                                       stride=(1, 1),
                                       nonlinearity=rectify,
                                       pad='same',
                                       W=nn.init.HeNormal(gain='relu')))

        # add shortcut connection
        if increase_dim:
            if projection:
                # projection shortcut (option B in paper)
                projection = batch_norm(ConvLayer(l,
                                                  num_filters=4*base_dim,
                                                  filter_size=(1, 1),
                                                  stride=(2, 2),
                                                  nonlinearity=None,
                                                  pad='same',
                                                  b=None))
                block = NonlinearityLayer(ElemwiseSumLayer([layer_3, projection]),
                                          nonlinearity=rectify)
            else:
                # identity shortcut (option A in paper)
                # we use a pooling layer to get identity with strides,
                # since identity layers with stride don't exist in Lasagne
                identity = PoolLayer(l, pool_size=1, stride=(2,2),
                                     mode='average_exc_pad')
                padding = PadLayer(identity, [4*base_dim,0,0], batch_ndim=1)
                block = NonlinearityLayer(ElemwiseSumLayer([layer_3, padding]),
                                          nonlinearity=rectify)

        else:
            block = NonlinearityLayer(ElemwiseSumLayer([layer_3, l]),
                                      nonlinearity=rectify)

        return block

    # Input of the network
    input_layer = InputLayer(shape=(batch_size,
                                    num_channels,
                                    input_height,
                                    input_width),
                             input_var=input_var)

    # Very first conv layer
    l = batch_norm(ConvLayer(input_layer,
                             num_filters=64,
                             filter_size=(7, 7),
                             stride=(2, 2),
                             nonlinearity=rectify,
                             pad='same',
                             W=nn.init.HeNormal(gain='relu')))

    # Maxpool layer
    l = MaxPoolLayer(l, pool_size=(3, 3), stride=(2, 2))

    # Convolove with 1x1 filter to match input dimension with the upcoming residual block
    l = batch_norm(ConvLayer(l,
                             num_filters=256,
                             filter_size=(1, 1),
                             stride=(1, 1),
                             nonlinearity=rectify,
                             pad='same',
                             W=nn.init.HeNormal(gain='relu')))

    ############# First residual blocks #############
    for _ in range(num_blocks[0] - 1):
        l = residual_block3(l, base_dim=64)

    ############# Second residual blocks ############
    # Increment Dimension
    l = residual_block3(l, base_dim=128, increase_dim=True, projection=True)
    for _ in range(num_blocks[1] - 1):
        l = residual_block3(l, base_dim=128)

    ############# Third residual blocks #############
    # Increment Dimension
    l = residual_block3(l, base_dim=256, increase_dim=True, projection=True)
    for _ in range(num_blocks[2] - 1):
        l = residual_block3(l, base_dim=256)

    ############# Fourth residual blocks #############
    # Increment Dimension
    l = residual_block3(l, base_dim=512, increase_dim=True, projection=True)
    for _ in range(num_blocks[2] - 1):
        l = residual_block3(l, base_dim=512)

    # Global pooling layer
    l = GlobalPoolLayer(l)

    # Softmax Layer
    softmax_layer = DenseLayer(l, num_units=output_dim,
                         W=nn.init.HeNormal(),
                         nonlinearity=softmax)

    return softmax_layer




