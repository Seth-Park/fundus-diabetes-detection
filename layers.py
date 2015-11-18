import lasagne
from theano import tensor as T
from theano.sandbox.cuda import dnn

import lasagne.layers.dnn
MaxPool2DLayer = lasagne.layers.dnn.MaxPool2DDNNLayer
Pool2DLayer = lasagne.layers.dnn.Pool2DDNNLayer

class RMSPoolLayer(Pool2DLayer):
    """Use RMS as pooling function.
    from https://github.com/benanne/kaggle-ndsb/blob/master/tmp_dnn.py
    """
    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0),
                 epsilon=1e12, **kwargs):
        super(RMSPoolLayer, self).__init__(incoming, pool_size, stride, pad, **kwargs)
        self.epsilon = epsilon
        del self.mode

    def get_output_for(self, input, *args, **kwargs):
        out = dnn.dnn_pool(T.sqr(input), self.pool_size, self.stride, 'average')
        return T.sqrt(out + self.epsilon)
