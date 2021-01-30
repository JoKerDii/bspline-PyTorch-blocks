# BSplineActivationFunc

import torch


class BSplineActivationFunc(torch.autograd.Function):
    """
    An Activation Function using B-spline transformation

    # Algorithm: https://www.duo.uio.no/bitstream/handle/10852/61162/thesisDouzette.pdf?sequence=1

    # Shape:
        - Input: (N, C, H, W)
        - Output: (N, C, H, W)

    # Usage:
        With BSplineActivation class
    """

    @staticmethod
    def forward(ctx, x, coefficients_vect, grid, zero_knot_indexes, size):

        x_clamped = x.clamp(
            min=-(grid.item() * (size // 2)), max=(grid.item() * (size // 2 - 1))
        )

        floored_x = torch.floor(x_clamped / grid)  # left coefficient
        fracs = x_clamped / grid - floored_x  # distance to left coefficient

        # indexes (in coefficients_vect) of the left coefficients
        indexes = (zero_knot_indexes.view(1, -1, 1, 1) + floored_x).long()
        ctx.save_for_backward(fracs, coefficients_vect, indexes, grid)

        # linear interpolation
        activation_output = coefficients_vect[indexes + 1] * fracs + coefficients_vect[
            indexes
        ] * (1 - fracs)
        return activation_output

    @staticmethod
    def backward(ctx, grad_out):
        fracs, coefficients_vect, indexes, grid = ctx.saved_tensors
        grad_x = (
            (coefficients_vect[indexes + 1] - coefficients_vect[indexes])
            / grid
            * grad_out
        )

        grad_coefficients_vect = torch.zeros_like(coefficients_vect)
        # right coefficients gradients
        grad_coefficients_vect.scatter_add_(
            0, indexes.view(-1) + 1, (fracs * grad_out).view(-1)
        )
        # left coefficients gradients
        grad_coefficients_vect.scatter_add_(
            0, indexes.view(-1), ((1 - fracs) * grad_out).view(-1)
        )
        return grad_x, grad_coefficients_vect, None, None, None
