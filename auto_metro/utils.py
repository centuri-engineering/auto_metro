import numpy as np

from scipy.fft import fftn, fftshift, ifftn, ifftshift
from scipy.signal import general_gaussian


def _fft(image):
    """shifted fft 2D
    """
    return fftshift(fftn(fftshift(image)))


def _ifft(im_fft):
    """shifted ifft 2D
    """
    return ifftshift(ifftn(ifftshift(im_fft)))


# apodImRect.m
def apodise(image, border, order=8):
    """
    Parameters
    ----------

    image: np.ndarray
    border: int, the size of the boreder in pixels

    Note
    ----
    The image is assumed to be of float datatype, no datatype management
    is performed.

    This is different from the original apodistation method,
    which multiplied the image borders by a quater of a sine.
    """
    # stackoverflow.com/questions/46211487/apodization-mask-for-fast-fourier-transforms-in-python
    nx, ny = image.shape
    # Define a general Gaussian in 2D as outer product of the function with itself
    window = np.outer(
        general_gaussian(nx, order, nx // 2 - border),
        general_gaussian(ny, order, ny // 2 - border),
    )
    ap_image = window * image

    return ap_image


def fft_dist(nx, ny):

    uu2, vv2 = np.meshgrid(np.fft.fftfreq(ny) ** 2, np.fft.fftfreq(nx) ** 2)
    dist = (uu2 + vv2) ** 0.5
    return dist  # / dist.sum()
