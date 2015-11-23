import numpy as np
from time import time
import pdb
import skimage
import matplotlib.pyplot as plt
import data_util
import pdb



DATA_DIR = 'converted'

files = data_util.get_image_files(DATA_DIR)

images = data_util.load_images(files)

MEAN = data_util.compute_mean(files)
STD = data_util.compute_std(files)

images_normalized = []
for img in images:
    img = img - MEAN[:, np.newaxis, np.newaxis]
    img = img / STD[:, np.newaxis, np.newaxis]
    images_normalized.append(img)

images_normalized = np.array(images_normalized)
original_augmented = data_util.parallel_augment(images)
normalized_augmented = data_util.parallel_augment(images_normalized)

original = images[3]
normalized = images_normalized[3]
original = original.transpose(1, 2, 0)
normalized = normalized.transpose(1, 2, 0)
oa = original_augmented[3]
oa = oa.transpose(1, 2, 0)
na = normalized_augmented[3]
na = na.transpose(1, 2, 0)

fig = plt.figure()
a = fig.add_subplot(2, 2, 1)
plt.imshow(original)
a.set_title('original')
a = fig.add_subplot(2, 2, 2)
plt.imshow(normalized)
a.set_title('normalized')
a = fig.add_subplot(2, 2, 3)
plt.imshow(oa)
a.set_title('original_augmented')
a = fig.add_subplot(2, 2, 4)
plt.imshow(na)
a.set_title('normalized_augmenged')
plt.show()


