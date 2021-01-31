# helper function

import numpy as np
import scipy.interpolate as si
import torch


def get_knots(start, end, n_bases=5, spline_order=3):
    """
    Arguments:
        x; torch.tensor of dim 1
    """
    x_range = end - start
    start = start - x_range * 0.001
    end = end + x_range * 0.001
    # mgcv annotation
    m = spline_order - 1
    nk = n_bases - m  # number of interior knots
    dknots = (end - start) / (nk - 1)
    knots = torch.linspace(
        start=start - dknots * (m + 1), end=end + dknots * (m + 1), steps=nk + 2 * m + 2
    )
    return knots.float()


def get_X_spline(x, knots, n_bases=5, spline_order=3, add_intercept=True):
    """
    Returns:
        torch.tensor of shape [len(x), n_bases + (add_intercept)]
    # BSpline formula
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html#scipy.interpolate.BSpline
    """
    cuda = False
    if x.is_cuda:
        cuda = True
    if len(x.shape) != 1:
        raise ValueError("x has to be 1 dimentional")
    tck = [knots, torch.zeros(n_bases), spline_order]
    X = torch.zeros([len(x), n_bases], dtype=torch.float32)
    x = x.cpu().numpy()  # TODO: tensor interpolation?
    for i in range(n_bases):
        vec = torch.zeros(n_bases, dtype=torch.float32)
        vec[i] = 1.0
        tck[1] = vec
        if cuda:
            X[:, i] = torch.from_numpy(si.splev(x, tck, der=0)).to(
                "cuda"
            )  # TODO: specify cuda number
        else:
            X[:, i] = torch.from_numpy(si.splev(x, tck, der=0))
    if add_intercept is True:
        ones = torch.ones_like(X[:, :1])
        X = torch.hstack([ones, X])
    return X


def get_S(n_bases=5, spline_order=3, add_intercept=True):
    # mvcv R-code
    # S<-diag(object$bs.dim);
    # if (m[2]) for (i in 1:m[2]) S <- diff(S)
    # object$S <- list(t(S)%*%S)  # get penalty
    # object$S[[1]] <- (object$S[[1]]+t(object$S[[1]]))/2 # exact symmetry

    S = np.identity(n_bases)
    m2 = spline_order - 1  # m[2] is the same as m[1] by default

    # m2 order differences
    for i in range(m2):
        S = np.diff(S, axis=0)  # same as diff() in R
    S = np.dot(S.T, S)
    S = (S + S.T) / 2  # exact symmetry
    if add_intercept is True:
        # S <- cbind(0, rbind(0, S)) # in R
        zeros = np.zeros_like(S[:1, :])
        S = np.vstack([zeros, S])
        zeros = np.zeros_like(S[:, :1])
        S = np.hstack([zeros, S])
    return S.astype(np.float32)


def _trunc(x, minval=None, maxval=None):
    """Truncate vector values to have values on range [minval, maxval]"""
    x = torch.clone(x)
    if minval != None:
        x[x < minval] = minval
    if maxval != None:
        x[x > maxval] = maxval
    return x


def encodeSplines(x, n_bases=5, spline_order=3, start=None, end=None, warn=True):
    """Function for the class `EncodeSplines`.
    Expansion by generating B-spline basis functions for each x
    and each n (spline-index) with `scipy.interpolate.splev`,
    based on the pre-placed equidistant knots on [start, end] range.

    # Arguments
        x: a torch.tensor of positions
        n_bases int: Number of spline bases.
        spline_order: 2 for quadratic, 3 for qubic splines
        start, end: range of values. If None, they are inferred from the data
        as minimum and maximum value.
        warn: Show warnings.

    # Returns
        `torch.tensor` of shape `(x.shape[0], x.shape[1], channels, n_bases)`
    """

    if len(x.shape) == 1:
        x = x.reshape((-1, 1))

    if start is None:
        start = torch.amin(x)  # should be np.nanmin
    else:
        if x.min() < start:
            if warn:
                print(
                    "WARNING, x.min() < start for some elements. Truncating them to start: x[x < start] = start"
                )
            x = _trunc(x, minval=start)
    if end is None:
        end = torch.amax(x)  # should be np.nanmax
    else:
        if x.max() > end:
            if warn:
                print(
                    "WARNING, x.max() > end for some elements. Truncating them to end: x[x > end] = end"
                )
            x = _trunc(x, maxval=end)
    bs = BSpline(start, end, n_bases=n_bases, spline_order=spline_order)

    # concatenate x to long
    assert len(x.shape) == 2
    n_rows = x.shape[0]
    n_cols = x.shape[1]

    x_long = x.reshape((-1,))

    # shape = (n_rows * n_cols, n_bases)
    x_feat = bs.predict(x_long, add_intercept=False)

    x_final = x_feat.reshape((n_rows, n_cols, n_bases))
    return x_final


def corr2d_stack(X, K):
    """iterate through the 0th dimension (channel dimension) of `X` and
    `K`. multiply them and stack together
    """
    out = torch.stack([torch.matmul(x, k) for x, k in zip(X, K)]).squeeze(-1)
    out = out.permute((1, 2, 0))
    return out
