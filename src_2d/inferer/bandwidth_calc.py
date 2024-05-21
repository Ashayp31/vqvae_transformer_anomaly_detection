import numpy as np
from scipy import fftpack
from scipy.optimize import brentq
import numbers
import itertools
import functools
import operator
import scipy

_use_Cython = False
try:
    FLOAT = scipy.float128
except AttributeError:
    FLOAT = np.float64

def cartesian(arrays):
    """
    Generate a cartesian product of input arrays.
    Adapted from:
        https://github.com/scikit-learn/scikit-learn/blob/
        master/sklearn/utils/extmath.py#L489
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    out = np.empty_like(ix, dtype=dtype)

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out


def autogrid(data, boundary_abs=3, num_points=None, boundary_rel=0.05):
    """
    Automatically select a grid if the user did not supply one.
    Input is (obs, dims), and so is ouput.
    number of grid : should be a power of two
    percentile : is how far out we go out
    Parameters
    ----------
    data : array-like
        Data with shape (obs, dims).
    boundary_abs: float
        How far out from boundary observations the grid goes in each dimension.
    num_points: int
        The number of points in the resulting grid (after cartesian product).
        Should be a number such that k**dims = `num_points`.
    boundary_rel: float
        How far out to go, relatively to max - min.
    Returns
    -------
    grid : array-like
        A grid of shape (obs, dims).
    Examples
    --------
    array([[-1., -1.],
           [-1.,  0.],
           [-1.,  1.],
           [ 0., -1.],
           [ 0.,  0.],
           [ 0.,  1.],
           [ 1., -1.],
           [ 1.,  0.],
           [ 1.,  1.]])
    array([[-0.5, -0.5],
           [-0.5,  0. ],
           [-0.5,  0.5],
           [ 0.5, -0.5],
           [ 0.5,  0. ],
           [ 0.5,  0.5]])
    """
    obs, dims = data.shape
    minimums, maximums = data.min(axis=0), data.max(axis=0)
    ranges = maximums - minimums

    if num_points is None:
        num_points = [int(np.power(1024, 1 / dims))] * dims
    elif isinstance(num_points, (numbers.Number,)):
        num_points = [num_points] * dims
    elif isinstance(num_points, (list, tuple)):
        pass
    else:
        msg = "`num_points` must be None, a number, or list/tuple for dims"
        raise TypeError(msg)

    if not len(num_points) == dims:
        raise ValueError("Number of points must be sequence matching dims.")

    list_of_grids = []

    generator = enumerate(zip(minimums, maximums, ranges, num_points))
    for i, (minimum, maximum, rang, points) in generator:
        assert points >= 2
        outside_borders = max(boundary_rel * rang, boundary_abs)
        list_of_grids.append(np.linspace(minimum - outside_borders, maximum + outside_borders, num=points))

    return cartesian(list_of_grids)


def linbin_numpy(data, grid_points, weights=None):
    """
    1D Linear binning using NumPy. Assigns weights to grid points from data.
    This function is fast for data sets upto approximately 1-10 million,
    it uses vectorized NumPy functions to perform linear binning. Takes around
    100 ms on 1 million data points, so not nearly as fast as the Cython
    implementation (10 ms).
    Parameters
    ----------
    data : array-like
        Must be of shape (obs,).
    grid_points : array-like
        Must be of shape (points,).
    weights : array-like
        Must be of shape (obs,).
    Examples
    --------
    True
    """
    # Convert the data and grid points
    data = np.asarray_chkfinite(data, dtype=float)
    grid_points = np.asarray_chkfinite(grid_points, dtype=float)
    assert len(data.shape) == 1
    assert len(grid_points.shape) == 1

    # Verify that the grid is equidistant
    diffs = np.diff(grid_points)
    assert np.allclose(np.ones_like(diffs) * diffs[0], diffs)

    if weights is None:
        weights = np.ones_like(data)

    weights = np.asarray_chkfinite(weights, dtype=float)
    weights = weights / np.sum(weights)

    if not len(data) == len(weights):
        raise ValueError("Length of data must match length of weights.")

    # Transform the data
    min_grid = np.min(grid_points)
    max_grid = np.max(grid_points)
    num_intervals = len(grid_points) - 1
    dx = (max_grid - min_grid) / num_intervals
    transformed_data = (data - min_grid) / dx

    # Compute the integral and fractional part of the data
    # The integral part is used for lookups, the fractional part is used
    # to weight the data
    fractional, integral = np.modf(transformed_data)
    integral = integral.astype(int)

    # Sort the integral values, and the fractional data and weights by
    # the same key. This lets us use binary search, which is faster
    # than using a mask in the the loop below
    indices_sorted = np.argsort(integral)
    integral = integral[indices_sorted]
    fractional = fractional[indices_sorted]
    weights = weights[indices_sorted]

    # Pre-compute these products, as they are used in the loop many times
    frac_weights = fractional * weights
    neg_frac_weights = weights - frac_weights

    # If the data is not a subset of the grid, the integral values will be
    # outside of the grid. To solve the problem, we filter these values away
    unique_integrals = np.unique(integral)
    unique_integrals = unique_integrals[(unique_integrals >= 0) & (unique_integrals <= len(grid_points))]

    result = np.asfarray(np.zeros(len(grid_points) + 1))
    for grid_point in unique_integrals:

        # Use binary search to find indices for the grid point
        # Then sum the data assigned to that grid point
        low_index = np.searchsorted(integral, grid_point, side="left")
        high_index = np.searchsorted(integral, grid_point, side="right")
        result[grid_point] += neg_frac_weights[low_index:high_index].sum()
        result[grid_point + 1] += frac_weights[low_index:high_index].sum()

    return result[:-1]


def linbin_Ndim_python(data, grid_points, weights=None):
    """
    d-dimensional linear binning. This is a slow, pure-Python function.
    Mainly used for testing purposes.
    With :math:`N` data points, and :math:`n` grid points in each dimension
    :math:`d`, the running time is :math:`O(N2^d)`. For each point the
    algorithm finds the nearest points, of which there are two in each
    dimension.
    Parameters
    ----------
    data : array-like
        The data must be of shape (obs, dims).
    grid_points : array-like
        Grid, where cartesian product is already performed.
    weights : array-like
        Must have shape (obs,).
    Examples
    --------
    >>> from KDEpy.utils import autogrid
    >>> grid_points = autogrid(np.array([[0, 0, 0]]), num_points=(3, 3, 3))
    >>> d = linbin_Ndim_python(np.array([[1.0, 0, 0]]), grid_points, None)
    """
    # Convert the data and grid points
    data = np.asarray_chkfinite(data, dtype=float)
    grid_points = np.asarray_chkfinite(grid_points, dtype=float)
    if weights is not None:
        weights = np.asarray_chkfinite(weights, dtype=float)
    else:
        # This is not efficient, but this function should just be correct
        # The faster algorithm is implemented in Cython
        weights = np.ones(data.shape[0])
    weights = weights / np.sum(weights)

    if (weights is not None) and (data.shape[0] != len(weights)):
        raise ValueError("Length of data must match length of weights.")

    obs_tot, dims = grid_points.shape

    # Compute the number of grid points for each dimension in the grid
    grid_num = (grid_points[:, i] for i in range(dims))
    grid_num = np.array(list(len(np.unique(g)) for g in grid_num))

    # Scale the data to the grid
    min_grid = np.min(grid_points, axis=0)
    max_grid = np.max(grid_points, axis=0)
    num_intervals = grid_num - 1  # Number of intervals
    dx = (max_grid - min_grid) / num_intervals
    data = (data - min_grid) / dx

    # Create results
    result = np.zeros(grid_points.shape[0], dtype=float)

    # Go through every data point
    for observation, weight in zip(data, weights):

        # Compute integer part and fractional part for every x_i
        # Compute relation to previous grid point, and next grid point
        int_frac = (
            (
                (int(coordinate), 1 - (coordinate % 1)),
                (int(coordinate) + 1, (coordinate % 1)),
            )
            for coordinate in observation
        )

        # Go through every cartesian product, i.e. every corner in the
        # hypercube grid points surrounding the observation
        for cart_prod in itertools.product(*int_frac):

            fractions = (frac for (integral, frac) in cart_prod)
            integrals = list(integral for (integral, frac) in cart_prod)
            # Find the index in the resulting array, compured by
            # x_1 * (g_2 * g_3 * g_4) + x_2 * (g_3 * g_4) + x_3 * (g_4) + x_4

            index = integrals[0]
            for j in range(1, dims):
                index = grid_num[j] * index + integrals[j]

            value = functools.reduce(operator.mul, fractions)
            result[index % obs_tot] += value * weight

    assert np.allclose(np.sum(result), 1)
    return result


def linbin_Ndim(data, grid_points, weights=None):
    """
    d-dimensional linear binning, when d >= 2.
    With :math:`N` data points, and :math:`n` grid points in each dimension
    :math:`d`, the running time is :math:`O(N2^d)`. For each point the
    algorithm finds the nearest points, of which there are two in each
    dimension. Approximately 200 times faster than pure Python implementation.
    Parameters
    ----------
    data : array-like
        The data must be of shape (obs, dims).
    grid_points : array-like
        Grid, where cartesian product is already performed.
    weights : array-like
        Must have shape (obs,).
    Examples
    --------
    >>> from KDEpy.utils import autogrid
    >>> grid_points = autogrid(np.array([[0, 0, 0]]), num_points=(3, 3, 3))
    >>> d = linbin_Ndim(np.array([[1.0, 0, 0]]), grid_points, None)
    """
    data_obs, data_dims = data.shape
    assert len(grid_points.shape) == 2
    assert data_dims >= 2

    # Convert the data and grid points
    data = np.asarray_chkfinite(data, dtype=float)
    grid_points = np.asarray_chkfinite(grid_points, dtype=float)
    if weights is not None:
        weights = np.asarray_chkfinite(weights, dtype=float)
        weights = weights / np.sum(weights)

    if (weights is not None) and (data.shape[0] != len(weights)):
        raise ValueError("Length of data must match length of weights.")

    obs_tot, dims = grid_points.shape

    # Compute the number of grid points for each dimension in the grid
    grid_num = (grid_points[:, i] for i in range(dims))
    grid_num = np.array(list(len(np.unique(g)) for g in grid_num))

    # Scale the data to the grid
    min_grid = np.min(grid_points, axis=0)
    max_grid = np.max(grid_points, axis=0)
    num_intervals = grid_num - 1
    dx = (max_grid - min_grid) / num_intervals
    data = (data - min_grid) / dx

    # Create results
    result = np.zeros(grid_points.shape[0], dtype=float)

    # Call the Cython implementation. Loops are unrolled if d=1 or d=2,
    # and if d >= 3 a more general routine is called. It's a bit slower since
    # the loops are not unrolled.

    # Weighted data has two specific routines
    if weights is not None:
        if data_dims >= 3:
            binary_flgs = cartesian(([0, 1],) * dims)
            result = cutils.iterate_data_ND_weighted(data, weights, result, grid_num, obs_tot, binary_flgs)
        else:
            result = cutils.iterate_data_2D_weighted(data, weights, result, grid_num, obs_tot)
        result = np.asarray_chkfinite(result, dtype=float)

    # Unweighted data has two specific routines too. This is because creating
    # uniform weights takes relatively long time. It's faster to have a
    # specialize routine for this case.
    else:
        if data_dims >= 3:
            binary_flgs = cartesian(([0, 1],) * dims)
            result = cutils.iterate_data_ND(data, result, grid_num, obs_tot, binary_flgs)
        else:
            result = cutils.iterate_data_2D(data, result, grid_num, obs_tot)
        result = np.asarray_chkfinite(result, dtype=float)
        result = result / data_obs

    assert np.allclose(np.sum(result), 1)
    return result


def linear_binning(data, grid_points, weights=None):
    """
    This wrapper function computes d-dimensional binning, very quickly.
    Computes binning by setting a linear grid and weighting points linearily
    by their distance to the grid points. In addition, weight asssociated with
    data points may be passed. Depending on whether or not weights are passed
    and the dimensionality of the data, specific sub-routines are called for
    fast evaluation.
    Parameters
    ----------
    data
        The data points.
    grid_points
        The number of points in the grid.
    weights
        The weights.
    Returns
    -------
    (grid, data)
        Data weighted at each grid point.
    Examples
    --------
    True
    """
    data = np.asarray_chkfinite(data, dtype=float)
    grid_points = np.asarray_chkfinite(grid_points, dtype=float)
    if weights is not None:
        weights = np.asarray_chkfinite(weights, dtype=float)

    # Make sure the dimensionality makes sense
    try:
        data_obs, data_dims = data.shape
    except ValueError:
        data_dims = 1

    try:
        grid_obs, grid_dims = grid_points.shape
    except ValueError:
        grid_dims = 1

    if not data_dims == grid_dims:
        raise ValueError("Shape of data and grid points must be the same.")

    if data_dims == 1:
        return linbin_numpy(data.ravel(), grid_points.ravel(), weights=weights)
    else:
        return linbin_Ndim(data, grid_points, weights=weights)

def _root(function, N, args):
    """
    Root finding algorithm. Based on MATLAB implementation by Botev et al.
    >>> # From the matlab code
    >>> ints = np.arange(1, 51)
    >>> ans = _root(_fixed_point, N=50, args=(50, ints, ints))
    >>> np.allclose(ans, 9.237610787616029e-05)
    True
    """
    # From the implementation by Botev, the original paper author
    # Rule of thumb of obtaining a feasible solution
    N = max(min(1050, N), 50)
    tol = 10e-12 + 0.01 * (N - 50) / 1000
    # While a solution is not found, increase the tolerance and try again
    found = 0
    while found == 0:
        try:
            # Other viable solvers include: [brentq, brenth, ridder, bisect]
            x, res = brentq(function, 0, tol, args=args, full_output=True, disp=False)
            found = 1 if res.converged else 0
        except ValueError:
            x = 0
            tol *= 2.0
            found = 0
        if x <= 0:
            found = 0

        # If the tolerance grows too large, minimize the function
        if tol >= 1:
            raise ValueError("Root finding did not converge. Need more data.")

    if not x > 0:
        raise ValueError("Root finding failed to find positive solution.")
    return x


def _fixed_point(t, N, I_sq, a2):
    r"""
    Compute the fixed point as described in the paper by Botev et al.
    .. math:
        t = \xi \gamma^{5}(t)
    Parameters
    ----------
    t : float
        Initial guess.
    N : int
        Number of data points.
    I_sq : array-like
        The numbers [1, 2, 9, 16, ...]
    a2 : array-like
        The DCT of the original data, divided by 2 and squared.
    Examples
    --------
    >>> # From the matlab code
    >>> ans = _fixed_point(0.01, 50, np.arange(1, 51), np.arange(1, 51))
    >>> assert np.allclose(ans, 0.0099076220293967618515)
    >>> # another
    >>> ans = _fixed_point(0.07, 25, np.arange(1, 11), np.arange(1, 11))
    >>> assert np.allclose(ans, 0.068342291525717486795)
    References
    ----------
     - Implementation by Daniel B. Smith, PhD, found at
       https://github.com/Daniel-B-Smith/KDE-for-SciPy/blob/master/kde.py
    """

    # This is important, as the powers might overflow if not done
    I_sq = np.asfarray(I_sq, dtype=FLOAT)
    a2 = np.asfarray(a2, dtype=FLOAT)

    # ell = 7 corresponds to the 5 steps recommended in the paper
    ell = 7

    # Fast evaluation of |f^l|^2 using the DCT, see Plancherel theorem
    f = (0.5) * np.pi ** (2 * ell) * np.sum(np.power(I_sq, ell) * a2 * np.exp(-I_sq * np.pi**2 * t))

    # Norm of a function, should never be negative
    if f <= 0:
        return -1
    for s in reversed(range(2, ell)):
        # This could also be formulated using the double factorial n!!,
        # but this is faster so and requires an import less

        # Step one: estimate t_s from |f^(s+1)|^2
        odd_numbers_prod = np.product(np.arange(1, 2 * s + 1, 2, dtype=FLOAT))
        K0 = odd_numbers_prod / np.sqrt(2 * np.pi)
        const = (1 + (1 / 2) ** (s + 1 / 2)) / 3
        time = np.power((2 * const * K0 / (N * f)), (2.0 / (3.0 + 2.0 * s)))

        # Step two: estimate |f^s| from t_s
        f = (0.5) * np.pi ** (2 * s) * np.sum(np.power(I_sq, s) * a2 * np.exp(-I_sq * np.pi**2 * time))

    # This is the minimizer of the AMISE
    t_opt = np.power(2 * N * np.sqrt(np.pi) * f, -2.0 / 5)

    # Return the difference between the original t and the optimal value
    return t - t_opt


def improved_sheather_jones(data, weights=None):
    """
    The Improved Sheater Jones (ISJ) algorithm from the paper by Botev et al.
    This algorithm computes the optimal bandwidth for a gaussian kernel,
    and works very well for bimodal data (unlike other rules). The
    disadvantage of this algorithm is longer computation time, and the fact
    that this implementation does not always converge if very few data
    points are supplied.
    Understanding this algorithm is difficult, see:
    https://books.google.no/books?id=Trj9HQ7G8TUC&pg=PA328&lpg=PA328&dq=
    sheather+jones+why+use+dct&source=bl&ots=1ETdKd_6EF&sig=jZk4R515GB1xsn-
    VZVnjr-JfjSI&hl=en&sa=X&ved=2ahUKEwi1_czNncTcAhVGhqYKHaPiBtcQ6AEwA3oEC
    AcQAQ#v=onepage&q=sheather%20jones%20why%20use%20dct&f=false
    Parameters
    ----------
    data: array-like
        The data points. Data must have shape (obs, 1).
    weights: array-like, optional
        One weight per data point. Must have shape (obs,). If None is
        passed, uniform weights are used.
    """
    obs, dims = data.shape
    if not dims == 1:
        raise ValueError("ISJ is only available for 1D data.")

    n = 2**10

    # weights <= 0 still affect calculations unless we remove them
    if weights is not None:
        data = data[weights > 0]
        weights = weights[weights > 0]

    # Setting `percentile` higher decreases the chance of overflow
    xmesh = autogrid(data, boundary_abs=6, num_points=n, boundary_rel=0.5)
    data = data.ravel()
    xmesh = xmesh.ravel()

    # Create an equidistant grid
    R = np.max(data) - np.min(data)
    # dx = R / (n - 1)
    data = data.ravel()
    N = len(np.unique(data))

    # Use linear binning to bin the data on an equidistant grid, this is a
    # prerequisite for using the FFT (evenly spaced samples)
    initial_data = linear_binning(data.reshape(-1, 1), xmesh, weights)
    assert np.allclose(initial_data.sum(), 1)

    # Compute the type 2 Discrete Cosine Transform (DCT) of the data
    a = fftpack.dct(initial_data)

    # Compute the bandwidth
    # The definition of a2 used here and in `_fixed_point` correspond to
    # the one cited in this issue:
    # https://github.com/tommyod/KDEpy/issues/95
    I_sq = np.power(np.arange(1, n, dtype=FLOAT), 2)
    a2 = a[1:] ** 2

    # Solve for the optimal (in the AMISE sense) t
    t_star = _root(_fixed_point, N, args=(N, I_sq, a2))

    # The remainder of the algorithm computes the actual density
    # estimate, but this function is only used to compute the
    # bandwidth, since the bandwidth may be used for other kernels
    # apart from the Gaussian kernel

    # Smooth the initial data using the computed optimal t
    # Multiplication in frequency domain is convolution
    # integers = np.arange(n, dtype=float)
    # a_t = a * np.exp(-integers**2 * np.pi ** 2 * t_star / 2)

    # Diving by 2 done because of the implementation of fftpack.idct
    # density = fftpack.idct(a_t) / (2 * R)

    # Due to overflow, some values might be smaller than zero, correct it
    # density[density < 0] = 0.
    bandwidth = np.sqrt(t_star) * R
    return bandwidth