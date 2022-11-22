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
    ξ =      (N-1)*h
    T = (Nsteps-1)*k
    r_0 = 0.25*ξ
    ε = 0.1*h
    T, R, Z = np.meshgrid(
        trange:=np.linspace(0, T, Nsteps),
        rrange:=np.linspace(0, ξ, M),
        zrange:=np.linspace(0, ξ, N),
        indexing='ij'
    )
    concentration_vectorized = np.vectorize(lambda t, r, z: concentration(t, r, z, h, ξ, r_0, ε, P=P))
    C_ana = concentration_vectorized(T, R, Z)
    C_ana[0] = np.zeros((N, M)) # Avoiding unnecessarily large values

    # Plot:
    sliders = plot_with_slider_2D(
        arrs=[C_FDM, C_ana],
        slider_labels=["t", "z"],
        plot_labels=["FDM", "Analytic"],
        axis_to_plot=2,
        xrange=zrange,
        xlabel="$r$",
        ylabel="c($t, r, z)$"
    )

    return sliders