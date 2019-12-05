# %matplotlib inline

import h5py, os
import numpy as np
from matplotlib import pyplot as plt

# Load the training data

file_path = '/home/kevinxu/Documents/NC2019MRI/train'

for num_samples, fname in enumerate(sorted(os.listdir(file_path))):
    subject_path = os.path.join(file_path, fname)
    with h5py.File(subject_path, 'a') as hf:
        print(f'The No. {num_samples+1} file {fname} key is {list(hf.keys())}')

# Peek at one sample of k-space data

sample_path = '/home/kevinxu/Documents/NC2019MRI/train/file1000000.h5'

with h5py.File(sample_path, 'r') as hf:
    volume_kspace = hf['kspace'][()]
    print('shape:', volume_kspace.shape)
    print('dtype:', volume_kspace.dtype)
    print('sum:', volume_kspace.sum())
    print('mean:', volume_kspace.mean())
    print('std:', volume_kspace.std())

# Show slices in the sample as k-space images

def show_slices(data, slice_nums, cmap=None):
    fig = plt.figure(figsize=(15,10))
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)
        plt.axis('off')
        plt.show()

show_slices(np.log(np.abs(volume_kspace) + 1e-9), [0, 10, 20, 30], cmap='gray')

# Show slices in the sample as real images

from functions import transforms as T

volume_kspace2 = T.to_tensor(volume_kspace)
volume_image = T.ifft2(volume_kspace2)
volume_image_abs = T.complex_abs(volume_image)

show_slices(volume_image_abs, [0, 10, 20, 30], cmap='gray')

# Simulate under-sample data

import torch
from functions.subsample import MaskFunc

mask_func0 = MaskFunc(center_fractions=[0.08], accelerations=[4])
mask_func1 = MaskFunc(center_fractions=[0.04], accelerations=[8])


## Coursework main purpose:
## We would like you to propose a machine/deep learning method that is 
## able to recontruct high quality images using large acceleration rates.