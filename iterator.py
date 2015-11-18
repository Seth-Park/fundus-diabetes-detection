import numpy as np
import theano
import theano.tensor as T
from nolearn.lasagne.base import BatchIterator

import data

class BatchIteratorAugmented(BatchIterator):
    def __init__(self, batch_size, transform=data.perturb_and_augment,
                 aug_params=data.no_augmentation_params, sigma=0.0, color_vec=None):
        self.batch_size = batch_size
        self.tf = transform
        self.aug_params = aug_params
        self.sigma = sigma
        self.color_vec = color_vec

    def __call__(self, X, y=None):
        self.X, self.y = X, y
        return self

    def __iter__(self):
        n_samples = self.X.shape[0]
        bs = self.batch_size
        for i in range((n_samples + bs - 1) // bs):
            sl = slice(i * bs, (i + 1) * bs)
            Xb = self.X[sl]
            if self.y is not None:
                yb = self.y[sl]
            else:
                yb = None
            yield self.transform(Xb, yb)

    def transform(self, Xb, yb):
        Xb = self.tf(Xb, 112, 112, aug_params=self.aug_params,
                     sigma=self.sigma, color_vec=self.color_vec)
        return Xb, yb



