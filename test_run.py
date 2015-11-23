import numpy as np
from time import time
import data
from matplotlib import pyplot as plt


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

#start = time()
#result = data.batch_perturb_and_augment(X, 500, 500, aug_params=aug_params, sigma=0.5)
#end = time()
#print("Processing without parallelization took {} seconds".format(end - start))

start = time()
result = data.parallel_perturb_and_augment(X, 500, 500, aug_params=aug_params, sigma=0.5)
end = time()
print("Processing with parallelization took {} seconds".format(end - start))

original_image = X[10]
original_image = original_image.transpose(1, 2, 0)

augmented_image = result[10]
augmented_image = augmented_image.transpose(1, 2, 0)

fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
plt.imshow(original_image)
a.set_title('Before')
a = fig.add_subplot(1, 2, 2)
plt.imshow(augmented_image)
a.set_title('After')
plt.show()
