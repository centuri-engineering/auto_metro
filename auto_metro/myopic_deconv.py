"""Myopic deconvolution algorithm from

Thibon, Louis, Ferréol Soulez, and Éric Thiébaut. _Fast automatic
myopic deconvolution of angiogram sequence_. In International
Symposium on Biomedical Imaging. Beijing, China,
2014. https://hal.archives-ouvertes.fr/hal-00914846.

See also:
https://www.nijboerzernike.nl/_PDF/JModOpt_ENZaberrationretrieval.pdf

"""
import numpy as np
from .zernike import zernike_nm


modes = [
    (0, 0),
    (2, -2),  # Oblique astigmatism
    (2, 2),  # Vertical astigmatism
    (3, -3),  # Vertical trefoil
    (3, -1),  # Vertical coma
    (3, 1),  # Horizontal coma
    (3, 3),  # Oblique trefoil
    (4, 0),  # Spherical
]


mode_names = [
    "Fundamental",
    "Oblique astigmatism",
    "Vertical astigmatism",
    "Vertical trefoil",
    "Vertical coma",
    "Horizontal coma",
    "Oblique trefoil",
    "Spherical",
]


def zernike_tf(nx, ny, pupil, mode_amps):

    rho, phi = np.meshgrid(np.linspace(-1, 1, ny), np.linspace(-1, 1, nx))
    W = np.zeros_like(rho)
    for (n, m), Anm in zip(modes, mode_amps):
        W += Anm * zernike_nm(rho * pupil, phi, n, m)
    W *= rho < pupil
    return W


def fft_dist(nx, ny):

    uu2, vv2 = np.meshgrid(
        np.fft.fftfreq(ny, d=1 / ny) ** 2, np.fft.fftfreq(nx, d=1 / nx) ** 2,
    )
    return (uu2 + vv2) ** 0.5


def power_law(nx, ny, alpha, beta):

    r = fft_dist(nx, ny) + np.finfo(float).eps
    return (10 ** alpha) * (r ** (np.abs(beta)))


def gen_max_likelihood(
    image_dsp, transfert_function, tf_params, prior_model, prior_params
):
    nx, ny = image_dsp.shape
    prior = prior_model(nx, ny, *prior_params)
    mtf = transfert_function(nx, ny, tf_params[0], tf_params[1:])
    mtf2 = np.abs(mtf) ** 2
    w = prior / (mtf2 + prior)
    # w = w[w > 0] why ?
    numer = (w * image_dsp).sum()
    denom = np.product(w.ravel()) / w.size
    return numer / denom


def opt_gml(params, image_dsp):

    gml = gen_max_likelihood(image_dsp, zernike_tf, params[2:], power_law, params[:2])
    return gml


# def GML(dataDSP, psfModel, psfParam, priorModel, priorParam):
#     """Compute the Generalize maximum likelihood given 'dataDSP' the data
#     powerspectrum, the PSF psfModel(dataDSP.shape, psfParam) and a prior
#     inverse powerspectrum of the object
#     `priorModel(dataDSP.shape,priorParam)`"""
#     q = priorModel(dataDSP.shape, priorParam)
#     mtf = fft(psfModel(dataDSP.shape, psfParam))
#     h2 = abs2(mtf)
#     w = q / (h2 + q)
#     num = np.sum(w * dataDSP)
#     w = w[w > 0]
#     denom = np.exp(np.sum(np.log(w)) / w.size)
#     return num / denom
