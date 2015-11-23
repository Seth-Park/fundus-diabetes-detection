import cv2
import os
from glob import glob
from PIL import Image
import numpy as np
from random import shuffle
import click
import pdb


@click.command()
@click.option('--datadir', default='testing2',
              show_default=True)
@click.option('--scale', default=300, show_default=True)
@click.option('--dest', default='preprocessed', show_default=True)
def main(datadir, scale, dest):
    files = get_image_files(datadir)
    for f in files:
        img = cv2.imread(f)
        img = scale_radius(img, scale)
        imgg = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), scale / 30), -4, 128)
        im = Image.fromarray(imgg)
        filename = f.split('/')[-1]
        im.save(os.path.join(dest, filename))


def scale_radius(img, scale):
    x = img[img.shape[0] / 2, :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    return cv2.resize(img, (0, 0), fx=s, fy=s)


def get_image_files(datadir, left_only=False, shuffle=False):
    files = glob('{}/*'.format(datadir))
    if left_only:
        files = [f for f in files if 'left' in f]
    if shuffle:
        return shuffle(files)
    return sorted(files)

if __name__ == '__main__':
    main()
