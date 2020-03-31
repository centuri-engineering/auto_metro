import numpy as np
from scipy.fft import fft2

from auto_metro.image_decorr import apodise, measure
from skimage import img_as_float
from skimage.io import imread


def test_measure():
    corti = img_as_float(imread("../samples/corti00.tif"))
    metadata = {"physicalSizeX": 0.3}
    snr, res = measure(corti, metadata).values()
    np.testing.assert_approx_equal(snr, 0.6, significant=2)
    np.testing.assert_approx_equal(res, 1.8, significant=2)


def test_apodise():
    image = np.random.random((800, 600))
    ap_image = apodise(image, 60)
    assert ap_image.shape == image.shape

    assert ap_image[:, 0].mean() < 1e-3
    assert ap_image[0, :].mean() < 1e-3
    np.testing.assert_array_almost_equal(
        ap_image[400:410, 300:310], image[400:410, 300:310], decimal=3
    )
