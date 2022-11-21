import numpy as np
import matplotlib.pyplot as plt
from Numeric.DiffusionFDM2D import DiffusionFDM2D
from Numeric.DiffusionFDM2DWithSink import DiffusionFDM2DWithSink


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
    system = DiffusionFDM2D(max_timestep=timestep+1, h=h, k=k, S=100, N=N, M=M, type=type)
    system.solve()
    system.plot(timestep, slider=True)
    return system


def test_DiffusionFDM2DWithSink(timestep: int, h: float, k: float, N: int = 10, M: int = 30, type: str = "backward") -> None:
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
    sink = np.zeros((N, M))
    sink[-1:, :int(0.25*M)] = 1
    system = DiffusionFDM2DWithSink(max_timestep=timestep+1, h=h, k=k, N=N, M=M, type=type, sink_fnc=sink)
    system.solve()
    print(f"Total fraction absorbed: {system.fraction_absorbed}")

    _, ax = plt.subplots(1, 2)
    system.plot(timestep, slider=True, ax=ax[0])
    system.plot_sinked(ax=ax[1])
    return system