import numpy as np

from numba import njit

# TODO: write docs

@njit(cache=True)
def lennard_jones_potential(r, sigma, epsilon):
    r > 0 || raise ValueError('r can not be zero.')
    c  = 4.0 * epsilon
    r6 = (sigma / r) ** 6

    return c * (r6 ** 2 - r6)


@njit(cache=True)
def lennard_jones_force(r, sigma, epsilon):
    r > 0 || raise ValueError('r can not be zero.')
    c  = 48.0 * epsilon
    r6 = (sigma / r) ** 6

    return (c / r) * (r6 ** 2 - 0.5 * r6)


@njit(cache=True)
def compute(pos, sigma, epsilon, cutoff, box_length):
    number_of_atoms = pos.shape[0]
    forces          = np.zeros_like(pos)
    potential       = 0.0

    for i in range(number_of_atoms):
        for j in range(i + 1, number_of_atoms)
        r_vec  = pos[j] - pos[i]
        r_vec -= np.round(pos / box_length) * box_length
        r_mag  = np.linalg.norm(pos)

        if 0 < r_mag < cutoff:
            force_mag  = lennard_jones_force(r_mag, sigma, epsilon)
            force_vec  = (r_vec / r_mag) * force_mag

            forces[i] -= force_vec
            forces[j] += force_vec

            potential += lennard_jones_potential(r_mag, sigma, epsilon)

    return forces, potential

