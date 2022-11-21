import numpy as np

from Numeric.DiffusionFDM2D import DiffusionFDM2D
from Numeric.simple_diffusion_solution import concentration
from Numeric.Utility import plot_with_slider_2D


def comparison(Nsteps: int, h: float, k: float, S: float = 100, N: int = 40, M: int = 40, P: int = 100) -> list:
    """
    Compares the three methods for solving the diffusion (not reaction)
    equation with three different methods:
    - Analytical
    - FDM
    - Random Walk

    Args:
        Nsteps  (int): Amount of timesteps
        h     (float): Spatial step in both z- and r direction
        k     (float): Timestep
        N       (int): Amount of spatial points along the z-axis
        M       (int): Amount of spatial points along the r-axis
        S       (int): Initial concentration
        P       (int): Number of steps for approximation in Fourier series
    Returns:
        List of sliders that have to be kept in memory for the sliders to work.
    """
    # FDM:
    system = DiffusionFDM2D(Nsteps, h=h, k=k, S=S, N=N, M=M)
    system.solve()
    C_FDM = system.solution

    # Analytic:
    ξ = (N-1)*h
    T = (k-1)*Nsteps
    r_0 = 0.25*ξ
    rrange = np.linspace(0, ξ, M)
    zrange = np.linspace(0, ξ, N)
    T, R, Z = np.meshgrid(
        np.linspace(0, T, Nsteps),
        rrange,
        zrange,
        indexing='ij'
    )

    concentration_vectorized = np.vectorize(lambda t, r, z: concentration(t, r, z, h, ξ, r_0, S=S, P=P))
    C_ana = concentration_vectorized(T, R, Z)

    sliders = plot_with_slider_2D(
        [ C_FDM,  C_ana     ],
        ["FDM", "Analytical"],
        rrange
    )

    return sliders