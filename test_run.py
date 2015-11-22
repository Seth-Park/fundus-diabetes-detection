import numpy as np
from time import time
import data


aug_params = {
    'zoom_range': (1 / 1.15, 1.15),
    'rotation_range': (0, 360),
    'shear_range': (0, 0),
    'translation_range': (-40, 40),
    'do_flip': True,
    'allow_stretch': True,
}


files = data.get_image_files('testing')
X = data.load_image(files)

print("Number of images: {}".format(len(X)))

start = time()
result = data.batch_perturb_and_augment(X, 500, 500, aug_params=aug_params, sigma=0.5)
end = time()
print("Processing without parallelization took {} seconds".format(end - start))

start = time()
result = data.parallel_perturb_and_augment(X, 500, 500, aug_params=aug_params, sigma=0.5)
end = time()
print("Processing with parallelization took {} seconds".format(end - start))

