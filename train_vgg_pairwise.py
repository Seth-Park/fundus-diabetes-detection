import os
import sys
import importlib
from time import time

import numpy as np
import pandas
import cPickle as pickle
import theano
import theano.tensor as T
import lasagne as nn
from sklearn.cross_validation import StratifiedShuffleSplit

from lasagne.objectives import categorical_crossentropy
from losses import (log_loss,
                    accuracy,
                    squared_loss,
                    quad_kappa,
                    quad_kappa_log_hybrid_loss,
                    quad_kappa_log_hybrid_loss_clipped)

from batch_iterator import *
from data_util import *
from model_util import *
import pdb

def load_model(mod):
    """
    Imports a predefined lasagne model.
    Calling build_model() will return the input layer and output layer
    of the model
    """
    return importlib.import_module(mod.replace('/', '.').split('.py')[0])

np.set_printoptions(precision=3)

# configs
input_size = 512
input_height, input_width = (input_size, input_size)
num_class = 5
num_channels = 3

# loss function config
l2_reg = 0.0002


# training configs
batch_size = 64
lr = theano.shared(np.array(0.005, dtype=theano.config.floatX))
lr_schedule = {2 : 0.005, 10 : 0.003, 20 : 0.001, 30 : 0.0005, 40: 0.0001 }
n_epochs = 30
momentum = 0.9
validate_every = 2000 # iterations
save_every = 10 # epochs

# paths
data_dir = "/nikel/dhpark/fundus/kaggle/original/training/train_medium"
label_file = "/nikel/dhpark/fundus/kaggle/original/training/trainLabels.csv"
#mean_file = ""
#model = "models/softmax_regression"
#model = "models/double_softmax"
model = "models/512x512_model"
#model = "models/vgg_bn_pairwise"
dst_path = "/nikel/dhpark/fundus_saved_weights/vgg_pairwise"
#dst_path = "/nikel/dhpark/fundus_saved_weights/multi_task_loss_oversampled"
#dst_path = "/nikel/dhpark/fundus_saved_weights/hybrid_loss"
#dst_path = "/nikel/dhpark/fundus_saved_weights/vgg_bn_pairwise"


# Load the model
x = T.tensor4('x')
y = T.imatrix('y')
input_layer, output_layer = load_model(model).build_model(x)

# Get batchiterator
#First load the files and split to train and validation set
#Then create a iterator using these
files = data_util.get_image_files(data_dir)
names = data_util.get_names(files)
labels = data_util.get_labels(names, label_file=label_file).astype(np.int32)
print('{} files loaded'.format(len(files)))

paired_files, paired_labels, merged_labels = data_util.pair_up(files, labels)

sss = StratifiedShuffleSplit(merged_labels, n_iter=1, test_size=0.1, random_state=123)
train_idx, valid_idx = next(iter(sss))

train_files = np.array(paired_files)[train_idx].tolist()  # some sort of hack..
train_labels = np.array(paired_labels)[train_idx].tolist()
merged_labels = merged_labels[train_idx]
n_train = len(train_files) * 2
print('Training set size: {}'.format(n_train))
coefs = [0, 7, 3, 22, 25]
train_files, train_labels = data_util.oversample_set_pairwise(train_files, train_labels, merged_labels, coefs)
n_train = len(train_files) * 2
assert len(train_files) == len(train_labels)
print('Oversampled set size: {}'.format(n_train))

#print('Computing mean...')
#mean = data_util.compute_mean_across_channels(train_files)
#mean.dump('mean.pkl')
#print('Computing std...')
#std = data_util.compute_mean_across_channels(train_files)
#std.dump('std.pkl')
mean = np.load('mean.pkl')
std = np.load('std.pkl')

valid_files = np.array(paired_files)[valid_idx].tolist()
valid_labels = np.array(paired_labels)[valid_idx].tolist()
n_valid = len(valid_files) * 2
print('Validation set size: {}'.format(n_valid))

train_iter = PairedBatchIterator(train_files,
                                 train_labels,
                                 batch_size,
                                 normalize=(mean, std),
                                 process_func=parallel_augment)
valid_iter = PairedBatchIterator(valid_files,
                                 valid_labels,
                                 batch_size,
                                 normalize=(mean, std),
                                 process_func=parallel_augment,
                                 testing=True)
# Transform batchiterator to a threaded iterator
train_iter = threaded_iterator(train_iter)
valid_iter = threaded_iterator(valid_iter)


# Construct loss function & accuracy
def multi_task_loss(y, t):
    softmax_predictions = categorical_crossentropy(y[:, :num_class], t)
    regress_predictions = squared_loss(y[:, -1], t)
    log_loss = softmax_predictions.mean()
    reg_loss = regress_predictions.mean()
    return log_loss, reg_loss, 0.75*log_loss + 0.25*reg_loss

def hybrid_loss(y, t):
    log_loss = categorical_crossentropy(y, t).mean()
    kappa_loss = quad_kappa_loss(y, t, y_pow=2)
    return kappa_loss + 0.5 * T.clip(log_loss, 0.6, 10 ** 3)


predictions = nn.layers.get_output(output_layer, deterministic=False)
log_loss = categorical_crossentropy(predictions, y).mean()
params = nn.layers.get_all_params(output_layer, regularizable=True)
regularization = sum(T.sum(p ** 2) for p in params)
train_loss = log_loss + l2_reg * regularization
train_accuracy = accuracy(predictions, y)
train_kappa = quad_kappa(predictions, y)


valid_predictions = nn.layers.get_output(output_layer, deterministic=True)
valid_loss = categorical_crossentropy(valid_predictions, y).mean()
valid_accuracy = accuracy(valid_predictions, y)
valid_kappa = quad_kappa(valid_predictions, y)

# Scale grads
all_params = nn.layers.get_all_params(output_layer, trainable=True)
all_grads = T.grad(train_loss, all_params)
#scaled_grads = nn.updates.total_norm_constraint(all_grads, max_norm=5, return_norm=False)

# Construct update
updates = nn.updates.nesterov_momentum(all_grads, all_params, learning_rate=lr, momentum=momentum)
#updates = nn.updates.adam(all_grads, all_params, learning_rate=0.0001)

# Compile functions
print('...compiling')
train = theano.function([x, y], (train_loss, train_accuracy, train_kappa), updates=updates, on_unused_input='warn')
validate = theano.function([x, y], (valid_loss, valid_accuracy, valid_kappa), on_unused_input='warn')

# Training Loop
print '...training'
sys.stdout.flush()

best_valid_loss_avg = -np.inf
best_iter = 0

start_time_all = time()

epoch = 0
n_train_batches = n_train / batch_size
n_valid_batches = n_valid / batch_size

while epoch < n_epochs:
    epoch = epoch + 1
    train_loss_list = []
    train_acc_list = []
    train_kappa_list = []
    print ''
    for minibatch_index in xrange(n_train_batches):
        iter = (epoch - 1) * n_train_batches + minibatch_index

        # fetch augmented batch
        #fetch_data_start = time()
        train_X, train_y = train_iter.next()
        #fetch_data_end = time()
        #print('fetching a batch took {} seconds'.format(fetch_data_end - fetch_data_start))

        # start iteration
        iter_start = time()
        t_loss, t_acc, t_kappa = train(train_X, train_y)
        train_loss_list.append(t_loss)
        train_acc_list.append(t_acc)
        train_kappa_list.append(t_kappa)
        if iter % 10 == 0:
            print('[epoch:%d/iter:%d] train_model(lr:%f) w/ minibatch(size:%d) train_loss: %.2f, kappa_score: %.2f, accuracy: %.2f %% - Elapsed time: %.2fs' % (epoch, iter + 1, lr.get_value(), batch_size, t_loss, t_kappa, t_acc * 100., time() - iter_start))
            sys.stdout.flush()

        # validate
        if (iter + 1) % validate_every == 0:
            valid_loss_list = []
            valid_acc_list = []
            valid_kappa_list = []
            for i in xrange(n_valid_batches):
                valid_X, valid_y = valid_iter.next()
                v_loss, v_acc, v_kappa = validate(valid_X, valid_y)
                valid_loss_list.append(v_loss)
                valid_acc_list.append(v_acc)
                valid_kappa_list.append(v_kappa)
            valid_loss_avg = np.mean(valid_loss_list)
            valid_acc_avg = np.mean(valid_acc_list)
            valid_kappa_avg = np.mean(valid_kappa_list)
            print('Validation --> [epoch:%d minibatch %i/%i, loss: %.2f, kappa_score: %.2f, accuracy: %.2f %%' % (epoch, minibatch_index + 1, n_train_batches, valid_loss_avg, valid_kappa_avg, valid_acc_avg * 100.))
            # update the best validation score
            if valid_loss_avg > best_valid_loss_avg:
                # save best validation score and iteration number
                best_valid_loss_avg = valid_loss_avg
                best_iter = iter

    # end of one epoch / report scores
    print('[epoch:%d] --> AVERAGE:\n train_loss: %.2f\ntrain_kappa: %.2f\ntrain_accuracy %.2f %%' % (epoch, np.mean(train_loss_list), np.mean(train_kappa_list), np.mean(train_acc_list) * 100.))

    # save model
    if epoch % save_every == 0:
        print('[save] %s' % os.path.join(dst_path, 'trained_net_epoch%d.pkl' % epoch))
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        write_params(output_layer, os.path.join(dst_path, 'trained_net_epoch%d.pkl' % epoch))

    if epoch in lr_schedule:
        lr.set_value(lr_schedule[epoch])

print('>> Training Complete')



