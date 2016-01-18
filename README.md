# fundus-diabetes-detection
[Kaggle Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection/)

## Data Preprocessing
The image size was resized to 256x256 and 512x512.
For very deep models like the recent Deep Residual Network by MSRA (http://arxiv.org/abs/1512.03385), 256x256 images were used due to GPU memory constraints. For other models, 512x512 images were used. 

## Oversampling
The data labels are highly unbalanced. For a more stable training, oversampling was done to balance the class labels (approximately uniform distribution). Oversampling should be stopped in the later part of the training to avoid overfitting the "rare" classes. 

## Train/Validation Split
90%/10% stratified random split using sklearn package.

## Multi-Threaded Data Loading and Realtime Data Augmentation (Multiprocessing)
Image batch is prefetched into a queue before being loaded onto GPU.
While being prefetched, data is normalized and randomly augmented (rotate, flip, scale, shear) toavoid overfitting. In test time, the images are only rotated randomly.

## Pairwise Merging of Features
There are two images per patient (left eye / right eye). To take this fact into account, the dense representations of the two eyes after the convolutional layers are concatenated before the last two fully-connected layers.

## Ideas Tried
- Dropout
- All Convolutional Network (http://arxiv.org/abs/1412.6806)
- Multitask Learning (Optimizing for both classification and regression task)
- Deep Residual Network (http://arxiv.org/abs/1512.03385)
- Maxout Network (http://arxiv.org/abs/1302.4389)
- Batch Normalization (http://arxiv.org/abs/1502.03167)
- Leaky ReLU (http://arxiv.org/abs/1505.00853)


