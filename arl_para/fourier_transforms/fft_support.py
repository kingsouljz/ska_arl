import numpy

def fft(a):
    """ Fourier transformation from image to grid space

    .. note::

        If there are four axes nthen the last outer axes are not transformed

    :param a: image in `lm` coordinate space
    :return: `uv` grid
    """
    if (len(a.shape) == 4):
        return numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.ifftshift(a, axes=[2, 3])), axes=[2, 3])
    else:
        return numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.ifftshift(a)))


def pad_mid(ff, npixel):
    """
    Pad a far field image with zeroes to make it the given size.

    Effectively as if we were multiplying with a box function of the
    original field's size, which is equivalent to a convolution with a
    sinc pattern in the uv-grid.

    .. note::

        If there are four axes then the last outer axes are not transformed

    :param ff: The input far field. Should be smaller than NxN.
    :param npixel:  The desired far field size

    """

    if len(ff.shape) == 4:
        nchan, npol, ny, nx = ff.shape
        cx = nx // 2
        cy = ny // 2
        if npixel == nx:
            return ff
        assert npixel > nx == ny
        pw = ((0, 0), (0, 0),
              (npixel // 2 - cy, npixel // 2 - cy),
              (npixel // 2 - cx, npixel // 2 - cx))
        return numpy.pad(ff,
                         pad_width=pw,
                         mode='constant',
                         constant_values=0.0)

    else:
        ny, nx = ff.shape
        cx = nx // 2
        cy = ny // 2
        if npixel == nx:
            return ff
        assert npixel > nx == ny
        pw = ((npixel // 2 - cy, npixel // 2 - cy),
              (npixel // 2 - cx, npixel // 2 - cx))
        return numpy.pad(ff,
                         pad_width=pw,
                         mode='constant',
                         constant_values=0.0)