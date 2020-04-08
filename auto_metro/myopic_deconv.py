"""Myopic deconvolution algorithm from

Thibon, Louis, Ferréol Soulez, and Éric Thiébaut. _Fast automatic
myopic deconvolution of angiogram sequence_. In International
Symposium on Biomedical Imaging. Beijing, China,
2014. https://hal.archives-ouvertes.fr/hal-00914846.

"""
import numpy as np
from scipy.optimize import minimize

from .utils import fft_dist, _fft
from .zernike import zernike_nm, MODES, MODE_NAMES


def zernike_tf(rho, phi, resolution, mode_amps, modes=None):
    """Zernike polynomials transfert function
    """
    pupil = resolution / np.pi
    if modes is None:
        modes = MODES
    W = np.zeros_like(rho)
    for (n, m), Anm in zip(modes, mode_amps):
        W += Anm * zernike_nm(rho * pupil, phi, n, m)
    W *= rho * pupil < 1.0
    return W / W.sum()


def power_law(dist, alpha, beta):
    """Power law regularization term
    """
    r = dist + np.finfo(float).eps
    return 10 ** alpha * (r ** (np.abs(beta)))


def estimate_psf(
    image, modes=None, initial_guess=None, fit_resolution=True, **min_kwargs
):
    """Estimates the parameter of the Zernike polynomial by a General Likelihood Maximum
    method described in  Thibon, Louis, Ferréol Soulez, and Éric Thiébaut. _Fast automatic
    myopic deconvolution of angiogram sequence_. In International
    Symposium on Biomedical Imaging. Beijing, China,
    2014. https://hal.archives-ouvertes.fr/hal-00914846.

    Parameters
    ----------
    image : np.ndarray
        the image to evaluate the PSF on
    modes : list of int pairs
        List of the modes of the Zernike polynomials to estimate, e.g
        [(2, -2), (2, 2), (4, 0)] will fit for both oblique and vertical astigmatism
        and spherical aberation
    initial_guess : dictionnary
        The initial estimate for the optimisation parameters
        with the following keys:
        alpha, beta : parameters of the prior
        resolution : the estimated image resolution
        (n, m) : amplitude of  Z_n^m
    fit_resolution : bool
        Whether to fit the resolution parameter (by changing the size of the transfer function pupil)
    **min_kwargs : all other keyword arguments are passed to scipy.optimize.minimize

    Returns
    -------
    deconv_params : dictionnary
        the optimized parameter, with the same keys as initial_guess


    See Also
    --------
    scipy.optimize.minimize The minimization algorithm
    """
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

    def gen_max_likelihood(prior_params, tf_params):
        """Equation 27 of Thibon et al. 2014
        """
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
        return numer / denom

    def opt_gml(params):
        prior_params = params[:2]
        if fit_resolution:
            tf_params = params[2:]
        else:
            tf_params = [initial["resolution"],] + list(params[2:])
        gml = gen_max_likelihood(prior_params, tf_params)
        return gml

    if fit_resolution:
        p0 = list(initial.values())
    else:
        p0 = [val for k, val in initial.items() if k != "resolution"]

    res = minimize(opt_gml, p0, **min_kwargs)

    if fit_resolution:
        alpha, beta, resolution, *amps = res.x
        deconv_params = {
            "alpha": alpha,
            "beta": beta,
            "resolution": resolution,
        }
    else:
        alpha, beta, *amps = res.x
        deconv_params = {
            "alpha": alpha,
            "beta": beta,
            "resolution": initial["resolution"],
        }
    deconv_params.update({mode: amp for mode, amp in zip(modes, amps)})
    return deconv_params
