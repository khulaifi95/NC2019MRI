
import numpy as np
import torch


class MaskFunc:
    """
    MaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the k-space data has N
    columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center corresponding to
           low-frequencies
        2. The other columns are selected uniformly at random with a probability equal to:
           prob = (N / acceleration - N_low_freqs) / (N - N_low_freqs).
    *This ensures that the expected number of columns selected is equal to (N / acceleration)*

    It is possible to use multiple center_fractions and accelerations, in which case one possible
    (center_fraction, acceleration) is chosen uniformly at random each time the MaskFunc object is
    called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04], then there
    is a 50% probability that 4-fold acceleration with 8% center fraction is selected and a 50%
    probability that 8-fold acceleration with 4% center fraction is selected.
    """

    def __init__(self, center_fractions, accelerations):
        """
        Args:
            center_fractions (List[float]): Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is chosen uniformly
                each time.

            accelerations (List[int]): Amount of under-sampling. This should have the same length
                as center_fractions. If multiple values are provided, then one of these is chosen
                uniformly each time. An acceleration of 4 retains 25% of the columns, but they may
                not be spaced evenly.
        """
        if len(center_fractions) != len(accelerations):
            raise ValueError('Number of center fractions should match number of accelerations')

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def __call__(self, shape, seed=None):
        """
        Args:
            shape (iterable[int]): The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last dimension.
            seed (int, optional): Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same shape.
        Returns:
            torch.Tensor: A mask of the specified shape.
        """
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
