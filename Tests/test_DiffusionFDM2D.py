from Numeric.DiffusionFDM2D import DiffusionFDM2D

def test_DiffusionFDM2D(timestep: int, h: float, k: float, N: int = 10, M: int = 30, type: str = "backward") -> None:
    """
    Testing the DiffusionFDM2D class by constructing and plotting.

    Args:
        Timestep (int): Timestep to plot
        h      (float): Spatial step in both z- and r direction
        k      (float): Timestep
        N        (int): Amount of spatial points along the z-axis
        M        (int): Amount of spatial points along the r-axis
        type  (string): Whether the method uses forward or backward FDM
    Returns:
        None
    """
    system = DiffusionFDM2D(max_timestep=timestep+1, h=h, k=k, S=100, N=N, M=M)
    system.solve()
    system.plot(timestep, slider=True)
    return system