import os
from multiprocessing.pool import Pool

import click
import numpy as np
import skimage
from skimage.transform import resize
from skimage.data import imread
from skimage.io import imsave

import data_util

def convert_size(fname, crop_size):
    img = imread(fname)
    result = resize(img, crop_size)
    return result


def convert(args):
    func, arg = args
    datadir, convert_dir, fname, crop_size = arg
    convert_fname = get_convert_fname(fname, datadir, convert_dir)

    if not os.path.exists(convert_fname):
        img = func(fname, crop_size)
        imsave(convert_fname, img)


def get_convert_fname(fname, datadir, convert_dir):
    return fname.replace(datadir, convert_dir)


@click.command()
@click.option('--datadir', default='preprocessed', show_default=True)
@click.option('--convert_dir', default='converted', show_default=True)
@click.option('--crop_size', default=(512, 512), show_default=True)
def main(datadir, convert_dir, crop_size):
    try:
        os.mkdir(convert_dir)
    except OSError:
        pass

    filenames = data_util.get_image_files(datadir)

    print('Resizing images in {} to {}'.format(datadir, convert_dir))

    n = len(filenames)

    batch_size = 500
    batches = n // batch_size + 1
    p = Pool()

    args = []

    for f in filenames:
        args.append((convert_size, (datadir, convert_dir, f, crop_size)))

    for i in range(batches):
        print('batch {:>2} / {}'.format(i + 1, batches))
        p.map(convert, args[i * batch_size : (i + 1) * batch_size])

    p.close()
    p.join()
    print('Done')

if __name__ == '__main__':
    main()



