"""Zernike polynomials
"""
import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from numpy.polynomial import Polynomial

ZERNIKE_POLYS = {}


def zernike_nm(rho, phi, n, m):
    """Returns the value of the Zernike polynomial of
    azymutal degree n and radial degree m at (rho, phi)
    polar coordinates.
    """
    if (n - m) % 2:
        return np.zeros_like(rho)

    if m >= 0:
        Z_nm = radial_poly(n, m)(rho) * np.cos(m * phi)
    else:
        Z_nm = radial_poly(n, -m)(rho) * np.sin(-m * phi)
    return Z_nm


def radial_poly(n, m):
    """Returns a numpy Polynomial instance for the
    given indices.
    """
    std_idx = standard_index(n, m)
    if std_idx in ZERNIKE_POLYS:
        return ZERNIKE_POLYS[std_idx]

    poly_coefs = np.zeros(n + 1)
    poly_coefs[n - 2 * np.arange((n - m) // 2 + 1)] = radial_factors(n, m)
    poly = Polynomial(poly_coefs, domain=[0, 1], window=[0, 1])
    ZERNIKE_POLYS[std_idx] = poly
    return poly


def radial_factors(n, m):
    """returns the radial pre-factor of the radial polynoms
    """
    r_nm = np.array(
        [
            (-1) ** k
            * factorial(n - k)
            / (factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k))
            for k in range((n - m) // 2 + 1)
        ]
    )
    return r_nm


def standard_index(n, m):
    """
    See:

    https://en.wikipedia.org/w/index.php?title=Zernike_polynomials#OSA/ANSI_standard_indices
    """
    return (n * (n + 2) + m) // 2


def rev_index(max_n):
    rev_idx = []
    for n in range(max_n + 1):
        for m in range(-n, n + 1):
            if (n - m) % 2:
                continue
            rev_idx.append((n, m))
    return rev_idx


def show_zern(n, m, grid_size):
    """Draws a polar representation of the Zernike polynomial
    of degree n, m with grid_size by grid_size samples.
    """
    rho, phi = np.meshgrid(
        np.linspace(0, 1, grid_size), np.linspace(-np.pi, np.pi, grid_size)
    )
    xx = rho * np.cos(phi)
    yy = rho * np.sin(phi)

    z_nm = zernike_nm(rho, phi, n, m)
    fig, ax = plt.subplots()
    ax.pcolormesh(xx, yy, z_nm)
    ax.set_aspect("equal")
    return fig, ax
