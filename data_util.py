import os
from glob import glob
from random import shuffle

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


def load_images(files):
    p = Pool()
    process = imread
    results = p.map(process, files)
    images = np.array(results, dtype=np.float32)
    images = images.transpose(0, 3, 1, 2)
    return images


def load_images_uint(files):
    p = Pool()
    process = imread
    results = p.map(process, files)
    images = np.array(results)
    images = images.transpose(0, 3, 1, 2)
    return images


def compute_mean(files, batch_size=512):
    ret = np.zeros(3)
    shape = None
    for i in range(0, len(files), batch_size):
        images = load_images(files[i : i + batch_size])
        shape = images.shape
        ret += images.sum(axis=(0, 2, 3))
    n = len(files) * shape[2] * shape[3]
    return (ret / n).astype(np.float32)


def compute_std(files, batch_size=512):
    s = np.zeros(3)
    s2 = np.zeros(3)
    shape = None
    for i in range(0, len(files), batch_size):
        images = np.array(load_images_uint(files[i : i + batch_size]), dtype=np.float64)
        shape = images.shape
        s += images.sum(axis=(0, 2, 3))
        s2 += np.power(images, 2).sum(axis=(0, 2, 3))
    n = len(files) * shape[2] * shape[3]
    var = (s2 - s**2.0 / n) / (n - 1)
    return np.sqrt(var)


def fast_warp(img, tf, mode='constant', order=0):
    m = tf.params
    t_img = np.zeros(img.shape, img.dtype)
    for i in range(t_img.shape[0]):
        t_img[i] = _warp_fast(img[i], m, mode=mode, order=order)
    return t_img


def build_augmentation_transform(test=False):
    # Need the following two lines to introduce randomness in multiprocessing
    # Somehow the processes share the same random numbers without these
    pid = mp.current_process()._identity[0]
    randst = np.random.mtrand.RandomState(pid)
    if not test:
        r = randst.uniform(-0.1, 0.1) # scale: increase or decrease up to 10%
        rotation = randst.uniform(0, 2 * 3.1415926535)
        skew = randst.uniform(-0.2, 0.2) + rotation
    else:
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

    print(homogenous_matrix)


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
    """
    Augment a single image.
    """
    augment = build_augmentation_transform(test)
    center, uncenter = build_center_uncenter_transforms(img.shape[1:])
    transform = uncenter + augment + center # move to center, augment, and then shift back
    img = fast_warp(img, transform, mode='constant', order=0)
    return img


def parallel_augment(images, test=False):
    """
    Augment multiple images through parallelization.
    """
    p = Pool()
    process = partial(augment, test=test)
    results = p.map(process, images)

    augmented_images = np.array(results, dtype=np.float32)
    return augmented_images



