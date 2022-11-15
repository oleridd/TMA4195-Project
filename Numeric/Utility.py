import numpy as np


def minmax(arr: np.ndarray):
    """
    Returns the min and max values of the given array.

    Args:
        arr (array)
    Returns:
        Tuple of the form (min(arr), max(arr))
    """
    return np.min(arr), np.max(arr)


def get_absorption_frac(sim, R: float, ε: float) -> float:
    """
    Given a RandomWalkSingleBoundary simulation, studies the pos variable
    to find the fraction of absorbed particles (particles that have reached)
    the upper boundary and are within a certain radius).

    Args:
        sim (RandomWalkSingleBoundary): RandomWalk simulation with boundary as defined in RandomWalkSingleBoundary.py
        R                      (float): Radius at boundary required for absorption
        ε                      (float): Acceptable distance from boundary to be absorbed (in z-direction)
    Returns:
        Fraction of absorbed particles
    """
    pos = sim.pos
    np.sum(
        pos[(pos[:, :, -1] < ε).logical_and(pos[:, :, 0]**2 + pos[:, :, 1]**2 < R**2)].flatten()
        )