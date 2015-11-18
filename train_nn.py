"""Training scirpt."""

import os
import click
import numpy as np
import importlib
import pdb

import data


@click.command()
@click.option('--model', default='models/128x128_pretrain.py', show_default=True,
              help='Path or name of model.')
@click.option('--datadir', default='/nikel/dhpark/fundus/kaggle/original/training/train_tiny',
              show_default=True, help='Path to the data source')
@click.option('--label', default='/nikel/dhpark/fundus/kaggle/original/training/trainLabels.csv',
              show_default=True, help='Path to the label source')
def main(model, datadir, label):

    files = data.get_image_files(datadir)
    names = data.get_names(files)
    y = data.get_labels(names, label_file=label).astype(np.float32)
    X = data.load_image(files)

    net = load_model(model).build_model()

    print("Training initiated...")
    net.fit(X, y)

def load_model(mod):
    return importlib.import_module(mod.replace('/', '.').split('.py')[0])

if __name__ == '__main__':
    main()



