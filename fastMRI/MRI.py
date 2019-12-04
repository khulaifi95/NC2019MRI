# %matplotlib inline

import h5py, os
import numpy as np
from matplotlib import pyplot as plt

# Load the training data.

file_path = '/home/kevinxu/Documents/NC2019MRI/train'

for num_samples, fname in enumerate(sorted(os.listdir(file_path))):
    subject_path = os.path.join(file_path, fname)
    with h5py.File(subject_path, 'a') as hf:
        print(f'The No. {num_samples} file {fname} key is {list(hf.keys())}')

# Peak at one sample of k-space data.

sample_path = '/home/kevinxu/Documents/NC2019MRI/train/file1000000.h5'

with h5py.File(sample_path,  "r") as hf:      # Read-only
    volume_kspace = hf['kspace'][()]
    print('shape:', volume_kspace.shape)
    print('dtye:', volume_kspace.dtype)
    print('sum:', volume_kspace.sum())
    print('mean:', volume_kspace.mean())
    print('std:', volume_kspace.std())

 # Show slices in the sample.


 def show_slices(data, slice_nums, cmap=None):
    fig = plt.figure(figsize=(15,10))
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)
        plt.axis('off')


 show_slices(np.log(np.abs(volume_kspace) + 1e-9), [5, 10, 20, 30], cmap='gray')
