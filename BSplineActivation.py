# BSplineActivation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from BSplineActivationFunc import BSplineActivationFunc


class BSplineActivation(nn.Module):
    """
    Applies a B-spline transformation to the input data

    # Algorithm
        https://www.duo.uio.no/bitstream/handle/10852/61162/thesisDouzette.pdf?sequence=1

    # Args
        mode: either 'linear' or 'conv', depending on the input of activation function
        size: odd integer, spline size
        grid: grid size
        num_activations: integer number of neurons 
        device: 'cpu' or 'gpu'
        dtype: tensor type

    # Main Methods
        **spline_grid_from_range**: compute spline grid spacing from desired one-side range.
        **init_zero_knot_indexes**: initialize indexes of zero knots of each activation.
        **reshape_forward**, **reshape_back**: make sure the shapes of input and output of the activation function are correct (2D or 4D tensor).
        **forward**: calculate the spline functions.
        **backward**: calculate the derivative of spline functions and coefficients.

    # Example:
        >>> m = BSplineActivation(num_activations=4)
        >>> input = torch.randn((4,4,5,5))
        >>> output = m(input)
    """

    def __init__(
        self,
        mode="conv",
        size=51,
        grid=0.1,
        num_activations=None,
        device="cpu",
        dtype=torch.float32,

    ):

        if mode not in ["conv", "linear"]:
            raise ValueError(
                'Mode should be either string "conv" or "linear".')
        if size % 2 == 0:
            raise ValueError("Size should be an odd integer.")
        if num_activations is None:
            raise ValueError("Need to provide int num_activations.")

        super().__init__()

        self.mode = mode
        self.size = size
        self.num_activations = num_activations
        self.device = device
        self.dtype = dtype
        self.grid = Tensor([grid]).to(**self.device_type)
        self.init_zero_knot_indexes()  # index of knot 0 for each filter/ neuron

        # tensor with locations of spline coefficients. size: (num_activations, size)
        grid_tensor = self.grid_tensor
        coefficients = torch.zeros_like(grid_tensor)

        half = self.num_activations // 2
        coefficients[0:half, :] = (grid_tensor[0:half, :]).abs()
        coefficients[half::, :] = F.softshrink(
            grid_tensor[half::, :], lambd=0.5)

        self.coefficients_vect = nn.Parameter(
            coefficients.contiguous().view(-1)
        )  # size: (num_activations*size)

        self.spline_size = [self.size]
        self.spline_range = 4
        self.spline_grid = [0.0] * len(self.spline_size)

        # a helper function
        def spline_grid_from_range(spline_size, range_=2, round_to=1e-6):
            """ Compute spline grid spacing from desired one-side range
            and the number of activation coefficients.
            """
            spline_grid = ((range_ / (spline_size//2)) // round_to) * round_to
            return spline_grid

        for i in range(len(self.spline_size)):
            self.spline_grid[i] = spline_grid_from_range(
                self.spline_size[i], self.spline_range
            )
        # len(self.spline_size) == 1

    @property
    def device_type(self):
        return dict(device=self.device, dtype=self.dtype)

    @property
    def grid_tensor(self):
        return self.get_grid_tensor(self.size, self.grid)

    def get_grid_tensor(self, size_, grid_):
        """Creates a 2D grid tensor of size (num_activations, size)
        with the positions of the B1 spline coefficients.
        """
        grid_arange = (
            torch.arange(-(size_ // 2), (size_ // 2) + 1)
            .to(**self.device_type)
            .mul(grid_)
        )
        grid_tensor = grid_arange.expand((self.num_activations, size_))
        return grid_tensor

    def init_zero_knot_indexes(self):
        """Initialize indexes of zero knots of each activation."""
        # self.zero_knot_indexes[i] gives index of knot 0 for filter/neuron_i.
        # size: (num_activations,)
        activation_arange = torch.arange(
            0, self.num_activations).to(**self.device_type)
        self.zero_knot_indexes = activation_arange * \
            self.size + (self.size // 2)

    def reshape_forward(self, input):
        input_size = input.size()
        if self.mode == "linear":
            if len(input_size) == 2:
                # one activation per conv channel
                x = input.view(
                    *input_size, 1, 1
                )  # transform to 4D size (N, num_units=num_activations, 1, 1)
            elif len(input_size) == 4:
                # one activation per conv output unit
                x = input.view(input_size[0], -1).unsqueeze(-1).unsqueeze(-1)
            else:
                raise ValueError(
                    f"input size is {len(input_size)}D but should be 2D or 4D..."
                )
        else:
            assert (
                len(input_size) == 4
            ), 'input to activation should be 4D (N, C, H, W) if mode="conv".'
            x = input
        return x

    def reshape_back(self, output, input_size):
        if self.mode == "linear":
            output = output.view(
                *input_size
            )  # transform back to 2D size (N, num_units)
        return output

    def forward(self, input):
        """
        Args:
            input : 2D/4D tensor
        """
        # len(self.spline_size) == 1:

        input_size = input.size()
        x = self.reshape_forward(input)
        assert x.size(
            1) == self.num_activations, "input.size(1) != num_activations."

        output = BSplineActivationFunc.apply(
            x, self.coefficients_vect_, self.grid, self.zero_knot_indexes, self.size
        )

        output = self.reshape_back(output, input_size)

        return output

    @property
    def coefficients_vect_(self):
        return self.coefficients_vect
