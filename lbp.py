import warnings

import numpy as np


def _local_binary_pattern(*args, **kwargs):  # real signature unknown
    """
    Gray scale and rotation invariant LBP (Local Binary Patterns).

        LBP is an invariant descriptor that can be used for texture classification.

        Parameters
        ----------
        image : (N, M) cnp.float64_t array
            Graylevel image.
        P : int
            Number of circularly symmetric neighbor set points (quantization of
            the angular space).
        R : float
            Radius of circle (spatial resolution of the operator).
        method : {'D', 'R', 'U', 'N', 'V'}
            Method to determine the pattern.

            * 'D': 'default'
            * 'R': 'ror'
            * 'U': 'uniform'
            * 'N': 'nri_uniform'
            * 'V': 'var'

        Returns
        -------
        output : (N, M) array
            LBP image.
    """
    pass

def check_nD(array, ndim, arg_name='image'):
    """
    Verify an array meets the desired ndims and array isn't empty.

    Parameters
    ----------
    array : array-like
        Input array to be validated
    ndim : int or iterable of ints
        Allowable ndim or ndims for the array.
    arg_name : str, optional
        The name of the array in the original function.

    """
    array = np.asanyarray(array)
    msg_incorrect_dim = "The parameter `%s` must be a %s-dimensional array"
    msg_empty_array = "The parameter `%s` cannot be an empty array"
    if isinstance(ndim, int):
        ndim = [ndim]
    if array.size == 0:
        raise ValueError(msg_empty_array % (arg_name))
    if array.ndim not in ndim:
        raise ValueError(
            msg_incorrect_dim % (arg_name, '-or-'.join([str(n) for n in ndim]))
        )



def local_binary_pattern(image, P, R, method='default'):
    """Compute the local binary patterns (LBP) of an image.

    LBP is a visual descriptor often used in texture classification.

    Parameters
    ----------
    image : (M, N) array
        2D grayscale image.
    P : int
        Number of circularly symmetric neighbor set points (quantization of
        the angular space).
    R : float
        Radius of circle (spatial resolution of the operator).
    method : str {'default', 'ror', 'uniform', 'nri_uniform', 'var'}, optional
        Method to determine the pattern:

        ``default``
            Original local binary pattern which is grayscale invariant but not
            rotation invariant.
        ``ror``
            Extension of default pattern which is grayscale invariant and
            rotation invariant.
        ``uniform``
            Uniform pattern which is grayscale invariant and rotation
            invariant, offering finer quantization of the angular space.
            For details, see [1]_.
        ``nri_uniform``
            Variant of uniform pattern which is grayscale invariant but not
            rotation invariant. For details, see [2]_ and [3]_.
        ``var``
            Variance of local image texture (related to contrast)
            which is rotation invariant but not grayscale invariant.

    Returns
    -------
    output : (M, N) array
        LBP image.

    References
    ----------
    .. [1] T. Ojala, M. Pietikainen, T. Maenpaa, "Multiresolution gray-scale
           and rotation invariant texture classification with local binary
           patterns", IEEE Transactions on Pattern Analysis and Machine
           Intelligence, vol. 24, no. 7, pp. 971-987, July 2002
           :DOI:`10.1109/TPAMI.2002.1017623`
    .. [2] T. Ahonen, A. Hadid and M. Pietikainen. "Face recognition with
           local binary patterns", in Proc. Eighth European Conf. Computer
           Vision, Prague, Czech Republic, May 11-14, 2004, pp. 469-481, 2004.
           http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.214.6851
           :DOI:`10.1007/978-3-540-24670-1_36`
    .. [3] T. Ahonen, A. Hadid and M. Pietikainen, "Face Description with
           Local Binary Patterns: Application to Face Recognition",
           IEEE Transactions on Pattern Analysis and Machine Intelligence,
           vol. 28, no. 12, pp. 2037-2041, Dec. 2006
           :DOI:`10.1109/TPAMI.2006.244`
    """
    check_nD(image, 2)

    methods = {
        'default': ord('D'),
        'ror': ord('R'),
        'uniform': ord('U'),
        'nri_uniform': ord('N'),
        'var': ord('V'),
    }
    if np.issubdtype(image.dtype, np.floating):
        warnings.warn(
            "Applying `local_binary_pattern` to floating-point images may "
            "give unexpected results when small numerical differences between "
            "adjacent pixels are present. It is recommended to use this "
            "function with images of integer dtype."
        )
    image = np.ascontiguousarray(image, dtype=np.float64)
    output = _local_binary_pattern(image, P, R, methods[method.lower()])
    return output