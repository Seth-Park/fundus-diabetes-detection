"""Training scirpt."""

import os
import click
import numpy as np
import importlib

import data


@click.command()
@click.option('--model', default='models/128x128_pretrain.py', show_default=True,
              help='Path or name of model.')
@click.option('--datadir', default='/nikel/dhpark/fundus/kaggle/original/training/train_tiny'

def load_model(mod):
    return importlib.import_module(mod.replace('/', '.').split('.py')[0])

def main():

    files = data.get_image_files(datadir)
    names = data.get_names(files)
    y = get_labels(names, label_file=os.path.join(datadir, 'trainLabels.csv')).astype(np.float32)
    X = load_image(files)

    net = load_model(model).build_model()

    print("Training initiated...")
    net.fit(X, y)

if __name__ == '__main__':
    main()



