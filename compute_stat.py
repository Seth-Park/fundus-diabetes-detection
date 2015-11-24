import numpy as np
import cPickle
import data_util

DATA_DIR = '/nikel/dhpark/fundus/kaggle/original/training/train_medium'

files = data_util.get_image_files(DATA_DIR)
mean = data_util.compute_mean_across_channels(files)
std = data_util.compute_std_across_channels(files)

print("computing done")
print("dumping...")
mean.dump("mean.dat")
std.dump("std.dat")

