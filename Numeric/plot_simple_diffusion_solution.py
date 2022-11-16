import numpy as np
import matplotlib.pyplot as plt

def plot_simple_diffusion_solution(t: float, h: float = 3, ξ: float = 25, r_0: float = 1, S: float = 1, N: int = 1000, M: int = 3000, P: int = 1000) -> None:
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
        np.linspace(0, h, N),
        np.linspace(0, ξ, M)
    )

    C = np.zeros((N, M))

    # This is a slow solution, but I am tired :'(
    for n in range(N):
        for m in range(M):
            C[n, m] = concentration(R[n, m], Z[n, m], t, h, ξ, r_0, S, P)
    
    ax = plt.figure().add_subplot(projection="3d")
    ax.plot(R, Z, C)



def concentration(r: float, z: float, t: float, h: float, ξ: float, r_0: float, S: float, P: int) -> float:
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
    Returns:
        The function value at the point r and z
    """
    π, sin, cos = np.pi, np.sin, np.cos

    m, n = np.arange(1, P), np.arange(P)
    s_m = sin(m*π*r_0/ξ)*cos(m*π*r/ξ)*np.exp(t*(m*π/ξ)**2) # NOTE: Should be /m here
    s_n =                cos(n*π*z/h)*np.exp(t*(n*π/h)**2)
    
    c = np.outer(s_m, s_n)*S/(π*r_0*h)
    c[0] *= 1/4     # These values corespond to κ for 2D Fourier
    c[0, :] *= 1/2
    c[:, 0] *= 1/2

    return np.sum(c)