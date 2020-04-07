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


def zernike_tf(rho, phi, resolution, mode_amps, modes=None):
    pupil = resolution / np.pi
    if modes is None:
        modes = MODES
    W = np.zeros_like(rho)
    for (n, m), Anm in zip(modes, mode_amps):
        W += Anm * zernike_nm(rho * pupil, phi, n, m)
    W *= rho * pupil < 1.0
    return W / W.sum()


def power_law(dist, alpha, beta):

    r = dist + np.finfo(float).eps
    return 10 ** alpha * (r ** (np.abs(beta)))


def deconvolution(image, modes=None, initial_guess=None, **min_kwargs):

    initial = {"alpha": 1.0, "beta": 2.0, "resolution": 2}
    for mode in modes:
        initial[mode] = 1e-6

    if initial_guess is not None:
        initial.update(initial_guess)

    image_dsp = np.abs(_fft(image)) ** 2
    image_dsp /= image_dsp.max()
    nx, ny = image_dsp.shape
    xx, yy = np.meshgrid(np.linspace(-1, 1, ny), np.linspace(-1, 1, nx))
    rho = (xx ** 2 + yy ** 2) ** 0.5
    phi = np.arctan2(yy, xx)
    dist = fft_dist(nx, ny)
    if modes is None:
        modes = [(2, -2), (2, 2), (4, 0)]
    costs = []

    def gen_max_likelihood(prior_params, tf_params):
        prior = power_law(dist, *prior_params)

        mtf = zernike_tf(
            rho,
            phi,
            resolution=tf_params[0],
            mode_amps=[1.0,] + tf_params[1:],
            modes=[(0, 0),] + modes,
        )
        mtf2 = np.abs(mtf) ** 2
        w = prior / (mtf2 + prior)
        numer = (w * image_dsp).sum()
        denom = np.exp(np.sum(np.log(w[w > 0])) / w.size)
        if denom == 0.0:
            print(prior_params)
            print(tf_params)

        return numer / denom

    def opt_gml(params):
        prior_params = params[:2]
        tf_params = params[2:]
        gml = gen_max_likelihood(prior_params, tf_params)
        costs.append(gml)
        return gml

    p0 = list(initial.values())
    res = minimize(opt_gml, p0, **min_kwargs)

    alpha, beta, resolution, *amps = res.x
    deconv_params = {
        "alpha": alpha,
        "beta": beta,
        "resolution": resolution,
    }
    deconv_params.update({mode: amp for mode, amp in zip(modes, amps)})
    return deconv_params, costs
