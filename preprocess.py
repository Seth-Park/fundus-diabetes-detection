import cv2
import os
from glob import glob
from PIL import Image
import numpy as np
from random import shuffle
import click
import pdb
from data_util import get_image_files


@click.command()
@click.option('--datadir', default='/nikel/dhpark/fundus/kaggle/original/training/train',
              show_default=True)
@click.option('--scale', default=300, show_default=True)
@click.option('--dest', default='/nikel/dhpark/fundus/kaggle/original/training/preprocessed',
              show_default=True)
def main(datadir, scale, dest):
    files = get_image_files(datadir)
    for f in files:
        try:
            img = cv2.imread(f)
            img = scale_radius(img, scale)
            imgg = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), scale / 30), -4, 128)
            im = Image.fromarray(imgg)
            filename = f.split('/')[-1]
            im.save(os.path.join(dest, filename))
        except Exception as e:
            print(e, "error in processing {}".format(f))
            continue
    print("done")



def scale_radius(img, scale):
    x = img[img.shape[0] / 2, :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / (r + 0.00001)
    return cv2.resize(img, (0, 0), fx=s, fy=s)


if __name__ == '__main__':
    main()
