import numpy as np
import matplotlib.pyplot as plt


def plot_simple_diffusion_solution(t: float, h: float = 3, ξ: float = 25, r_0: float = 1, S: float = 4, N: int = 1000, M: int = 3000, P: int = 1000) -> None:
    """
    Plots the concentration function below in a 3D projection.

    Args:
        t   (float): Time
        h   (float): Synapse height
        ξ   (float): Upper bound for r
        r_0 (float): Radius of neurotransmitter outlet
        S   (float): Integral of function (conserved quantity of the system)
        N     (int): Gridpoints in the z-direction
        M     (int): Gridpoints in the r-direction
        P     (int): Partial sum quantity, such that the complexity for each gridpoint is O(P²)
    Returns:
        None
    """
    R, Z = np.meshgrid(
        np.linspace(0, ξ, M),
        np.linspace(0, h, N)
    )

    concentration_vectorized = np.vectorize(lambda r, z: concentration(r, z, t, h, ξ, r_0, S, P))
    C = concentration_vectorized(R, Z)
    
    ax = plt.figure().add_subplot(projection="3d")
    ax.plot_surface(R, Z, C)
    ax.set_xlabel("$r$")
    ax.set_ylabel("$z$")


def concentration(r: float, z: float, t: float, h: float, ξ: float, r_0: float, S: float, P: int, κ: float = 8e-7) -> float:
    """
    Solution to the diffusion equation on the form
    
    dc/dt = ∇²c
    
    with Neumann boundary conditions

     dc/dz(z=h) = 0
    -dc/dz(z=0) = 0
     dc/dr(r=ξ) = 0
    -dc/dr(r=0) = 0

    Args:
        r   (float): r-coordinate
        z   (float): z-coordinate
        t   (float): Time
        h   (float): Synapse height
        ξ   (float): Upper bound for r
        r_0 (float): Radius of neurotransmitter outlet
        S   (float): Integral of function (conserved quantity of the system)
        P     (int): Partial sum quantity, such that the complexity for each gridpoint is O(P²)
        κ   (float): Diffusion coefficient [m^2/s]
    Returns:
        The function value at the point r and z
    """
    π, sin, cos = np.pi, np.sin, np.cos

    m, n = np.arange(P), np.arange(P)
    s_m = cos(m*π*r/ξ)*np.exp(-t*(m*π/ξ)**2)
    s_n = cos(n*π*z/h)*np.exp(-t*(n*π/h)**2)

    s_m[0 ] *= π*r_0/ξ
    s_m[1:] *= sin(m[1:]*π*r_0/ξ)/m[1:]

    c = np.outer(s_m, s_n)*κ*S/(4*π*r_0*h)
    c[0   ] *= 4 # These values corespond to κ for 2D Fourier
    c[0, :] *= 2
    c[:, 0] *= 2

    return np.sum(c)