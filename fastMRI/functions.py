import numpy as np
import torch


##
###############################subsample###################################
##
class MaskFunc:
	def __init__(self, center_fractions, accelerations):

        if len(center_fractions) != len(accelerations):
            raise ValueError('Number of center fractions should match number of accelerations')

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def __call__(self, shape, seed=None):

        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')

        self.rng.seed(seed)     # ensure the same mask is generated
        num_cols = shape[-2]    # number of columns of the mask; using -2 to double-check the shape

        # Randomly make a choice from inputs
        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        # Create the mask
        num_low_freqs = int(round(num_cols * center_fraction))  # number of columns of center fractions that kept in the mask
        prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)   # percentage of columns to be masked
        mask = self.rng.uniform(size=num_cols) < prob   # return a boolean distribution
        pad = (num_cols - num_low_freqs + 1) // 2   # padding needed on two sides
        mask[pad:pad + num_low_freqs] = True    # don't mask out the center fraction
        # mask[pad:pad + num_low_freqs - 1] = True

        # Solution to uncertainty of k-fold under-sampling strategy
        # num_low_freqs = int(round(num_cols * center_fraction))
        # prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
        # pad = (num_cols - num_low_freqs + 1) // 2
        # mask_high = self.rng.uniform(size=num_cols - num_low_freqs) < prob
        # mask_low = np.full(num_low_freqs, True)
        # mask = np.concatenate([mask_high[:pad], mask_low, mask_high[pad + num_low_freqs:]])
        
        # Reshape the mask
        mask_shape = [1 for _ in shape] # return a array of 1 as input shape
        mask_shape[-2] = num_cols   # ensure mask match the original length 
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask


##
###############################transforms###################################
##

# 1)
def tensor_to_complex_np(data):

    data = data.numpy()
    return data[..., 0] + 1j * data[..., 1]


# 2)
def to_tensor(data):
	to_tensor(data):

    if np.iscomplexobj(data):
    	data = np.stack((data.real, data.imag), axis=-1)
    return torch..from_numpy(data)


# 3)
def apply_mask(data, mask_func, seed=None):

    shape = np.array(data.shape)
    shape[:-3] = 1
    mask = mask_func(shape, seed)
    return torch.where(mask == 0, torch.Tensor([0]), data), mask


# 4)
def fft2(data):

    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = forch.fft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


# 5)
def ifft2(data):

    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.ifft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


# 6)
def complex_abs(data):
	assert data.size(-1) == 2
	return (data ** 2).sum(dim=-1).sqrt()


# 7)
def center_crop(data, shape):

    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :]


# 8)
def complex_center_crop(data, shape):

    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :]


# 9)
def normalize(data, mean, stddev, eps=0.):

    return (data - mean) / (stddev + eps)


# 10)
def normalize_instance(data, eps=0.):

    mean = data.mean()
    std = data.std()
    return normailize(data, mean, std, eps), mean, std


# Helper functions

def roll(x, shift, dim):

    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):

    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):

    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)