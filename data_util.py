import os
from glob import glob
from random import shuffle
from time import time

import numpy as np
import pandas as pd
from PIL import Image
import skimage
from skimage.data import imread
import skimage.transform
from skimage.transform._warps_cy import _warp_fast

import multiprocessing as mp
from multiprocessing.pool import Pool
from functools import partial
from math import sin, cos

import pdb


def get_image_files(datadir, left_only=False, shuffle=False):
    files = glob('{}/*'.format(datadir))
    if left_only:
        files = [f for f in files if 'left' in f]
    if shuffle:
        return shuffle(files)
    return sorted(files)

def pair_up(files, labels):
    """
    Assuming that files are sorted,
    return a list of tuples with files of the same patient paired together
    and the corresponding one-hot-encoded labels.
    """
    paired_files = []
    paired_labels = []
    merged_labels = []
    one_hot_encoded = one_hot(labels)
    while len(files) != 0:
        paired_files.append((files[0], files[1]))
        index = np.random.randint(2)
        merged_labels.append(labels[index])
        paired_labels.append((one_hot_encoded[0], one_hot_encoded[1]))
        files = files[2:]
        labels = labels[2:]
        one_hot_encoded = one_hot_encoded[2:]
    return paired_files, paired_labels, np.array(merged_labels)


def get_names(files):
    return [os.path.basename(x).split('.')[0] for x in files]


def get_labels(names, labels=None, label_file='data/trainLabels.csv', per_patient=False):
    if labels is None:
        labels = pd.read_csv(label_file,
                             index_col=0).loc[names].values.flatten()

    if per_patient:
        left = np.array(['left' in n for n in names])
        return np.vstack([labels[left], labels[~left]]).T
    else:
        return labels

def one_hot(labels):
    identity = np.eye(max(labels) + 1)
    return identity[labels].astype(np.int32)

def load_images(files):
    p = Pool()
    process = imread
    results = p.map(process, files)
    #images = np.array(results, dtype=np.float32)
    p.close()
    p.join()
    images = np.array(results)
    images = images.transpose(0, 3, 1, 2)
    return images


def load_images_uint(files):
    p = Pool()
    process = imread
    results = p.map(process, files)
    p.close()
    p.join()
    images = np.array(results)
    images = images.transpose(0, 3, 1, 2)
    return images


def compute_mean_across_channels(files, batch_size=512):
    ret = np.zeros(3)
    shape = None
    for i in range(0, len(files), batch_size):
        print('processing from {}'.format(i))
        images = load_images(files[i : i + batch_size])
        shape = images.shape
        ret += images.sum(axis=(0, 2, 3))
    n = len(files) * shape[2] * shape[3]
    return (ret / n).astype(np.float32)


def compute_std_across_channels(files, batch_size=512):
    s = np.zeros(3)
    s2 = np.zeros(3)
    shape = None
    for i in range(0, len(files), batch_size):
        print('processing from {}'.format(i))
        images = np.array(load_images_uint(files[i : i + batch_size]), dtype=np.float64)
        shape = images.shape
        s += images.sum(axis=(0, 2, 3))
        s2 += np.power(images, 2).sum(axis=(0, 2, 3))
    n = len(files) * shape[2] * shape[3]
    var = (s2 - s**2.0 / n) / (n - 1)
    return np.sqrt(var).astype(np.float32)


def compute_stat_pixel(files, batch_size=512):
    dummy_img = load_images(files[0])
    shape = dummy_img.shape[1:]
    mean = np.zeros(shape)
    batches = []

    for i in range(0, len(files), batch_size):
        images = load_images(files[i : i + batch_size])
        batches.append(images)
        mean += images.sum(axis=0)
    n = len(files)
    mean = (mean / n).astype(np.float32)

    std = np.zeros(shape)
    for b in batches:
        std += ((b - mean) ** 2).sum(axis=0)
    std = np.sqrt(std / (n - 1)).astype(np.float32)
    return mean, std


def fast_warp(img, tf, mode='constant', order=0):
    m = tf.params
    t_img = np.zeros(img.shape, img.dtype)
    for i in range(t_img.shape[0]):
        t_img[i] = _warp_fast(img[i], m, mode=mode, order=order)
    return t_img


def build_augmentation_transform(test=False):
    pid = mp.current_process()._identity[0]
    randst = np.random.mtrand.RandomState(pid + int(time() % 3877))
    if not test:
        r = randst.uniform(-0.1, 0.1)  # scale
        rotation = randst.uniform(0, 2 * 3.1415926535)
        skew = randst.uniform(-0.2, 0.2) + rotation
    else: # only rotate randomly during test time
        r = 0
        rotation = randst.uniform(0, 2 * 3.1415926535)
        skew = rotation

    homogenous_matrix = np.zeros((3, 3))
    c00 = (1 + r) * cos(rotation)
    c10 = (1 + r) * sin(rotation)
    c01 = -(1 - r) * sin(skew)
    c11 = (1 - r) * cos(skew)

    # flip every other time
    if randst.randint(0, 2) == 0:
        c00 *= -1
        c10 *= -1

    homogenous_matrix[0][0] = c00
    homogenous_matrix[1][0] = c10
    homogenous_matrix[0][1] = c01
    homogenous_matrix[1][1] = c11
    homogenous_matrix[2][2] = 1


    transform = skimage.transform.AffineTransform(homogenous_matrix)
    return transform


def build_center_uncenter_transforms(image_shape):
    """
    These are used to ensure that zooming and rotation happens around the center of the image.
    Use these transforms to center and uncenter the image around such a transform.
    """

    # need to swap rows and cols here apparently! confusing!
    center_shift = np.array([image_shape[1], image_shape[0]]) / 2.0 - 0.5
    tform_uncenter = skimage.transform.SimilarityTransform(translation=-center_shift)
    tform_center = skimage.transform.SimilarityTransform(translation=center_shift)
    return tform_center, tform_uncenter


def augment(img, test=False):
    augment = build_augmentation_transform(test)
    center, uncenter = build_center_uncenter_transforms(img.shape[1:])
    transform = uncenter + augment + center
    img = fast_warp(img, transform, mode='constant', order=0)
    return img


def parallel_augment(images, normalize=None, test=False):
    if normalize is not None:
        mean, std = normalize
        images = images - mean[:, np.newaxis, np.newaxis] # assuming channel-wise normalization
        images = images / std[:, np.newaxis, np.newaxis]

    p = Pool()
    process = partial(augment, test=test)
    results = p.map(process, images)
    p.close()
    p.join()
    augmented_images = np.array(results, dtype=np.float32)
    return augmented_images

def oversample_set(files, labels, coefs):
    """
    files: list of filenames in the train set
    labels: the corresponding labels for the files
    coefs: list of oversampling ratio for each class
    Code modified from github.com/JeffreyDF.`
    """

    train_1 = list(np.where(np.apply_along_axis(
        lambda x: 1 == x,
        0,
        labels))[0])
    train_2 = list(np.where(np.apply_along_axis(
        lambda x: 2 == x,
        0,
        labels))[0])
    train_3 = list(np.where(np.apply_along_axis(
        lambda x: 3 == x,
        0,
        labels))[0])
    train_4 = list(np.where(np.apply_along_axis(
        lambda x: 4 == x,
        0,
        labels))[0])

    print(len(train_1), len(train_2), len(train_3), len(train_4))
    X_oversample = list(files)
    X_oversample += list(np.array(files)[coefs[1] * train_1])
    X_oversample += list(np.array(files)[coefs[2] * train_2])
    X_oversample += list(np.array(files)[coefs[3] * train_3])
    X_oversample += list(np.array(files)[coefs[4] * train_4])

    y_oversample = np.array(labels)
    y_oversample = np.hstack([y_oversample, labels[coefs[1] * train_1]])
    y_oversample = np.hstack([y_oversample, labels[coefs[2] * train_2]])
    y_oversample = np.hstack([y_oversample, labels[coefs[3] * train_3]])
    y_oversample = np.hstack([y_oversample, labels[coefs[4] * train_4]])

    return X_oversample, y_oversample


def oversample_set_pairwise(files, labels, merged, coefs):
    """
    files: list of paired filenames in the train set
    labels: the corresponding label pairs for the file pairs
    merged: merged labels
    coefs: list of oversampling ratio for each class
    Code modified from github.com/JeffreyDF.`
    """

    train_1 = list(np.where(np.apply_along_axis(
        lambda x: 1 == x,
        0,
        merged))[0])
    train_2 = list(np.where(np.apply_along_axis(
        lambda x: 2 == x,
        0,
        merged))[0])
    train_3 = list(np.where(np.apply_along_axis(
        lambda x: 3 == x,
        0,
        merged))[0])
    train_4 = list(np.where(np.apply_along_axis(
        lambda x: 4 == x,
        0,
        merged))[0])

    print(len(train_1), len(train_2), len(train_3), len(train_4))
    X_oversample = list(files)
    X_oversample += list(np.array(files)[coefs[1] * train_1])
    X_oversample += list(np.array(files)[coefs[2] * train_2])
    X_oversample += list(np.array(files)[coefs[3] * train_3])
    X_oversample += list(np.array(files)[coefs[4] * train_4])

    y_oversample = list(labels)
    y_oversample += list(np.array(labels)[coefs[1] * train_1])
    y_oversample += list(np.array(labels)[coefs[2] * train_2])
    y_oversample += list(np.array(labels)[coefs[3] * train_3])
    y_oversample += list(np.array(labels)[coefs[4] * train_4])

    return X_oversample, y_oversample




