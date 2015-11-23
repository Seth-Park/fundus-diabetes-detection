"""IO and data augmentation.

The code for data augmentation originally comes from
https://github.com/sveitser/kaggle_diabetic
"""
from __future__ import division, print_function
from collections import Counter
import os
from glob import glob
from random import shuffle

import numpy as np
import pandas as pd
from PIL import Image
import skimage
import skimage.transform
from skimage.transform._warps_cy import _warp_fast
from sklearn.utils import shuffle
from sklearn import cross_validation
from multiprocessing.pool import Pool
from functools import partial

RANDOM_STATE = 9
FEATURE_DIR = 'data/features'

# channel standard deviations
#STD = np.array([70.53946096, 51.71475228, 43.03428563], dtype=np.float32)
STD = np.array([70.53946096, 51.71475228, 43.03428563])

# channel means
#MEAN = np.array([108.64628601, 75.86886597, 54.34005737], dtype=np.float32)
MEAN = np.array([108.64628601, 75.86886597, 54.34005737])

# set of resampling weights that yields balanced classes
BALANCE_WEIGHTS = np.array([1.3609453700116234,  14.378223495702006,
                            6.637566137566138, 40.235967926689575,
                            49.612994350282484])

# for color augmentation, computed with make_pca.py
#U = np.array([[-0.56543481, 0.71983482, 0.40240142],
#              [-0.5989477, -0.02304967, -0.80036049],
#              [-0.56694071, -0.6935729, 0.44423429]] ,dtype=np.float32)
#EV = np.array([1.65513492, 0.48450358, 0.1565086], dtype=np.float32)
U = np.array([[-0.56543481, 0.71983482, 0.40240142],
              [-0.5989477, -0.02304967, -0.80036049],
              [-0.56694071, -0.6935729, 0.44423429]])
EV = np.array([1.65513492, 0.48450358, 0.1565086])

no_augmentation_params = {
    'zoom_range': (1.0, 1.0),
    'rotation_range': (0, 0),
    'shear_range': (0, 0),
    'translation_range': (0, 0),
    'do_flip': False,
    'allow_stretch': False,
}


def fast_warp(img, tf, output_shape, mode='constant', order=0):
    """
    This wrapper function is faster than skimage.transform.warp
    """
    m = tf.params
    t_img = np.zeros((img.shape[0], ) + output_shape, img.dtype)
    for i in range(t_img.shape[0]):
        t_img[i] = _warp_fast(img[i], m, output_shape=output_shape,
                              mode=mode, order=order)
    return t_img


def build_centering_transform(image_shape, target_shape):
    rows, cols = image_shape
    trows, tcols = target_shape
    shift_x = (cols - tcols) / 2.0
    shift_y = (rows - trows) / 2.0
    return skimage.transform.SimilarityTransform(translation=(shift_x, shift_y))


def build_center_uncenter_transforms(image_shape):
    """
    These are used to ensure that zooming and rotation happens around the center of the image.
    Use these transforms to center and uncenter the image around such a transform.
    """
    center_shift = np.array([image_shape[1], image_shape[0]]) / 2.0 - 0.5 # need to swap rows and cols here apparently! confusing!
    tform_uncenter = skimage.transform.SimilarityTransform(translation=-center_shift)
    tform_center = skimage.transform.SimilarityTransform(translation=center_shift)
    return tform_center, tform_uncenter


def build_augmentation_transform(zoom=(1.0, 1.0), rotation=0, shear=0, translation=(0, 0), flip=False):
    if flip:
        shear += 180
        rotation += 180
        # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
        # So after that we rotate it another 180 degrees to get just the flip.

    tform_augment = skimage.transform.AffineTransform(scale=(1/zoom[0], 1/zoom[1]), rotation=np.deg2rad(rotation), shear=np.deg2rad(shear), translation=translation)
    return tform_augment


def random_perturbation_transform(zoom_range, rotation_range, shear_range, translation_range, do_flip=True, allow_stretch=False, rng=np.random):
    shift_x = rng.uniform(*translation_range)
    shift_y = rng.uniform(*translation_range)
    translation = (shift_x, shift_y)

    rotation = rng.uniform(*rotation_range)
    shear = rng.uniform(*shear_range)

    if do_flip:
        flip = (rng.randint(2) > 0) # flip half of the time
    else:
        flip = False

    # random zoom
    log_zoom_range = [np.log(z) for z in zoom_range]
    if isinstance(allow_stretch, float):
        log_stretch_range = [-np.log(allow_stretch), np.log(allow_stretch)]
        zoom = np.exp(rng.uniform(*log_zoom_range))
        stretch = np.exp(rng.uniform(*log_stretch_range))
        zoom_x = zoom * stretch
        zoom_y = zoom / stretch
    elif allow_stretch is True: # avoid bugs, f.e. when it is an integer
        zoom_x = np.exp(rng.uniform(*log_zoom_range))
        zoom_y = np.exp(rng.uniform(*log_zoom_range))
    else:
        zoom_x = zoom_y = np.exp(rng.uniform(*log_zoom_range))
    # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.

    return build_augmentation_transform((zoom_x, zoom_y), rotation, shear, translation, flip)

def perturb(img, augmentation_params, target_shape, rng=np.random):
    # # DEBUG: draw a border to see where the image ends up
    # img[0, :] = 0.5
    # img[-1, :] = 0.5
    # img[:, 0] = 0.5
    # img[:, -1] = 0.5
    shape = img.shape[1:]
    tform_centering = build_centering_transform(shape, target_shape)
    tform_center, tform_uncenter = build_center_uncenter_transforms(shape)
    tform_augment = random_perturbation_transform(rng=rng, **augmentation_params)
    tform_augment = tform_uncenter + tform_augment + tform_center # shift to center, augment, shift back (for the rotation/shearing)
    return fast_warp(img, tform_centering + tform_augment, # apply the augmentation and center it
                     output_shape=target_shape,
                     mode='constant')


# for test-time augmentation
def perturb_fixed(img, tform_augment, target_shape=(50, 50)):
    shape = img.shape[1:]
    tform_centering = build_centering_transform(shape, target_shape)
    tform_center, tform_uncenter = build_center_uncenter_transforms(shape)
    tform_augment = tform_uncenter + tform_augment + tform_center # shift to center, augment, shift back (for the rotation/shearing)
    return fast_warp(img, tform_centering + tform_augment,
                     output_shape=target_shape, mode='constant')


def load_perturbed(fname):
    img = util.load_image(fname).astype(np.float32)
    return perturb(img)


def augment_color(img, sigma=0.1, color_vec=None):

    if color_vec is None:
        if not sigma > 0.0:
            color_vec = np.zeros(3, dtype=np.float32)
        else:
            color_vec = np.random.normal(0.0, sigma, 3)

    alpha = color_vec.astype(np.float32) * EV
    noise = np.dot(U, alpha.T)
    return img + noise[:, np.newaxis, np.newaxis]

def single_perturb_augment(img, w, h, aug_params=no_augmentation_params, sigma=0.0):
    img = perturb(img, augmentation_params=aug_params, target_shape=(w, h))
    img = img - MEAN[:, np.newaxis, np.newaxis]
    img = img / STD[:, np.newaxis, np.newaxis]
    #np.subtract(img, MEAN[:, np.newaxis, np.newaxis], out=img)
    #np.divide(img, STD[:, np.newaxis, np.newaxis], out=img)
    img = augment_color(img, sigma=sigma)
    return img


def batch_perturb_and_augment(img, w, h, aug_params=no_augmentation_params, sigma=0.0):
    """Transform and augment image whose output shape is (w, h).
    Defualt arguments return non augmented image of shape (w, h).
    To generate a random augmentation, specify aug_params and sigma.
    """
    images = []
    for im in img:
        im = single_perturb_augment(im, w=w, h=w, aug_params=aug_params, sigma=sigma)
        images.append(im)

    perturbed_images = np.array(images, dtype=np.float32)
    return perturbed_images


def parallel_perturb_and_augment(img, w, h, aug_params=no_augmentation_params, sigma=0.0):
    images = img
    p = Pool()
    process = partial(single_perturb_augment, w=w, h=h, aug_params=aug_params, sigma=sigma)
    results = p.map(process, images)

    perturbed_images = np.array(results, dtype=np.float32)
    return perturbed_images


def compute_mean(files, batch_size=128):
    """Load images in files in batches and compute mean."""
    m = np.zeros(3)
    for i in range(0, len(files), batch_size):
        images = load_image(files[i : i + batch_size])
        m += images.sum(axis=(0, 2, 3))
    return (m / len(files)).astype(np.float32)


def std(files, batch_size=128):
    s = np.zeros(3)
    s2 = np.zeros(3)
    shape = None
    for i in range(0, len(files), batch_size):
        print("done with {:>3} / {} images".format(i, len(files)))
        images = np.array(load_image_uint(files[i : i + batch_size]),
                          dtype=np.float64)
        shape = images.shape
        s += images.sum(axis=(0, 2, 3))
        s2 += np.power(images, 2).sum(axis=(0, 2, 3))
    n = len(files) * shape[2] * shape[3]
    var = (s2 - s**2.0 / n) / (n - 1)
    return np.sqrt(var)


def get_labels(names, labels=None, label_file='data/trainLabels.csv',
               per_patient=False):

    if labels is None:
        labels = pd.read_csv(label_file,
                             index_col=0).loc[names].values.flatten()

    if per_patient:
        left = np.array(['left' in n for n in names])
        return np.vstack([labels[left], labels[~left]]).T
    else:
        return labels


def get_image_files(datadir, left_only=False, shuffle=False):
    fs = glob('{}/*'.format(datadir))
    if left_only:
        fs = [f for f in fs if 'left' in f]
    if shuffle:
        return np.array(shuffle(fs))
    return np.array(sorted(fs))


def get_names(files):
    return [os.path.basename(x).split('.')[0] for x in files]


def load_image(fname):
    if isinstance(fname, basestring):
        #return np.array(Image.open(fname), dtype=np.float32).transpose(2, 1, 0)
        return np.array(Image.open(fname)).transpose(2, 1, 0)
    else:
        return np.array([load_image(f) for f in fname])


def split_indices(files, labels, test_size=0.1, random_state=RANDOM_STATE):
    names = get_names(files)
    labels = get_labels(names, per_patient=True)
    spl = cross_validation.StratifiedShuffleSplit(labels[:, 0],
                                                  test_size=test_size,
                                                  random_state=random_state,
                                                  n_iter=1)
    tr, te = next(iter(spl))
    tr = np.hstack([tr * 2, tr * 2 + 1])
    te = np.hstack([te * 2, te * 2 + 1])
    return tr, te


def split(files, labels, test_size=0.1, random_state=RANDOM_STATE):
    train, test = split_indices(files, labels, test_size, random_state)
    return files[train], files[test], labels[train], labels[test]



def load_features(fnames, test=False):

    if test:
        fnames = [os.path.join(os.path.dirname(f),
                               os.path.basename(f).replace('train', 'test'))
                  for f in fnames]

    data = [np.load(f) for f in fnames]
    data = [X.reshape([X.shape[0], -1]) for X in data]
    return np.hstack(data)


def parse_blend_config(cnf):
    return {run: [os.path.join(FEATURE_DIR, f) for f in files]
            for run, files in cnf.items()}

