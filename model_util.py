import cPickle as pickle
import os

import lasagne as nn

def read_params(model, filename):
    """
    Unpickles and loads parameters into a Lasagne model.
    """
    params = None
    with open(filename, 'r') as f:
        params = pickle.load(f)
    nn.layers.set_all_param_values(model, params)

def write_params(model, filename):
    """
    Pickles the parameters within a Lasagne model.
    """
    params = nn.layers.get_all_param_values(model)
    with open(filename, 'w') as f:
        pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
