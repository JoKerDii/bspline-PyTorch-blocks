# BSpline
import numpy as np

from helper import *


class BSpline:
    """
    Class for computing the B-spline funcions b_i(x)
    and constructing the penality matrix S.

    # Args
        start: float or int; start of the region
        end: float or int; end of the region
        n_bases: int; number of spline bases
        spline_order: int; spline order

    # Methods
        - **getS(add_intercept=False)** - Get the penalty matrix S
              - Args
                     - **add_intercept**: bool. If true, intercept column is added to the returned matrix.
              - Returns
                     - `np.array`, of shape `(n_bases + add_intercept, n_bases + add_intercept)`
        - **predict(x, add_intercept=False)** - For some x, predict the bn(x) for each base
              - Args
                     - **x**: np.array; Vector of dimension 1
                     - **add_intercept**: bool; If True, intercept column is added to the to the final array
              - Returns
                     - `torch.tensor`, of shape `(len(x), n_bases + (add_intercept))`
    """

    def __init__(self, start=0, end=1, n_bases=10, spline_order=3):

        self.start = start
        self.end = end
        self.n_bases = n_bases
        self.spline_order = spline_order

        self.knots = get_knots(self.start, self.end, self.n_bases, self.spline_order)

        self.S = get_S(self.n_bases, self.spline_order, add_intercept=False)

    def __repr__(self):
        return "BSpline(start={0}, end={1}, n_bases={2}, spline_order={3})".format(
            self.start, self.end, self.n_bases, self.spline_order
        )

    def getS(self, add_intercept=False):
        """Get the penalty matrix S
        Returns:
            torch.tensor, of shape (n_bases + add_intercept, n_bases + add_intercept)
        """
        S = self.S
        if add_intercept is True:
            # S <- cbind(0, rbind(0, S)) # in R
            zeros = np.zeros_like(S[:1, :])
            S = np.vstack([zeros, S])

            zeros = np.zeros_like(S[:, :1])
            S = np.hstack([zeros, S])
        return S

    def predict(self, x, add_intercept=False):
        """For some x, predict the bn(x) for each base
        Args:
            x: torch.tensor
            add_intercept: bool; should we add the intercept to the final array
        Returns:
            torch.tensor, of shape (len(x), n_bases + (add_intercept))
        """
        # sanity check
        if x.min() < self.start:
            raise Warning("x.min() < self.start")
        if x.max() > self.end:
            raise Warning("x.max() > self.end")

        return get_X_spline(
            x=x,
            knots=self.knots,
            n_bases=self.n_bases,
            spline_order=self.spline_order,
            add_intercept=add_intercept,
        )

    def get_config(self):
        return {
            "start": self.start,
            "end": self.end,
            "n_bases": self.n_bases,
            "spline_order": self.spline_order,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
