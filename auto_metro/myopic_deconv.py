"""Myopic deconvolution algorithm from

Thibon, Louis, Ferréol Soulez, and Éric Thiébaut. _Fast automatic
myopic deconvolution of angiogram sequence_. In International
Symposium on Biomedical Imaging. Beijing, China,
2014. https://hal.archives-ouvertes.fr/hal-00914846.

See also:
https://www.nijboerzernike.nl/_PDF/JModOpt_ENZaberrationretrieval.pdf

"""
import numpy as np
from scipy.optimize import minimize

from .utils import fft_dist, _fft
from .zernike import zernike_nm, MODES, MODE_NAMES


def zernike_tf(rho, phi, pupil, mode_amps, modes=None):
    if modes is None:
        modes = MODES
    W = np.zeros_like(rho)
    for (n, m), Anm in zip(modes, mode_amps):
        W += Anm * zernike_nm(rho * pupil, phi, n, m)
    W *= rho * pupil < 1.0
    return W


def power_law(dist, alpha, beta):

    r = dist + np.finfo(float).eps
    return (10 ** alpha) * (r ** (np.abs(beta)))


def deconvolution(image, modes=None, resolution=1, alpha=10, beta=1, **min_kwargs):

    image_dsp = np.abs(_fft(image)) ** 2
    nx, ny = image_dsp.shape
    xx, yy = np.meshgrid(np.linspace(-1, 1, ny), np.linspace(-1, 1, nx))
    rho = (xx ** 2 + yy ** 2) ** 0.5
    phi = np.arctan2(yy, xx)
    dist = fft_dist(nx, ny)
    if modes is None:
        modes = [(2, -2), (2, 2), (4, 0)]
    pupil = resolution / np.pi
    p0 = [alpha, beta, pupil] + [1e-3,] * (len(modes))
    costs = []

    def gen_max_likelihood(tf_params, prior_params):
        prior = power_law(dist, *prior_params)
        mtf = zernike_tf(
            rho,
            phi,
            pupil=tf_params[0],
            mode_amps=[1.0,] + tf_params[1:],
            modes=[(0, 0),] + modes,
        )
        mtf2 = np.abs(mtf) ** 2
        w = prior / (mtf2 + prior)
        # w = w[w > 0] why ?
        numer = (w * image_dsp).sum()
        denom = np.product(w) ** (1 / w.size)
        return numer / denom

    def opt_gml(params):
        gml = gen_max_likelihood(params[2:], params[:2])
        costs.append(gml)
        return gml

    res = minimize(opt_gml, p0, **min_kwargs)
    print(res.message)
    alpha, beta, pupil, *amps = res.x
    deconv_params = {
        "α": alpha,
        "β": beta,
        "resolution": np.pi * pupil,
    }
    deconv_params.update({mode: amp for mode, amp in zip(modes, amps)})
    return deconv_params, costs
