import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from Numeric.DiffusionFDM2D import DiffusionFDM2D
from Numeric.Utility import plot_with_silder


class DiffusionFDM2DWithSink(DiffusionFDM2D):

    """
    Finite difference method for the diffusion equation in two dimensions.
    Adds a sink "profile", removing particles by some rate given by
    f: ℝ² -> ℝ²
    f ∈ [0, 1] (coresponding to the fraction of particles absorbed)
    """

    def __init__(self, max_timestep: int, h: float, k: float, κ: float = 8e-7, S: float = 500, N: int = 10, M: int = 30, type: str = "backward", IC: np.ndarray = None, sink_fnc: np.ndarray = None) -> None:
        """
        Args:
            max_timestep (int): The maximum timestep
            h          (float): Spatial step in both z- and r direction
            k          (float): Timestep
            κ          (float): Diffusion coefficient [m^2/s]
            S          (float): Conserved integral over c (the solution)
            N            (int): Amount of spatial points along the z-axis
            M            (int): Amount of spatial points along the r-axis
            type      (string): Whether the method uses forward or backward FDM
            IC     (NxM Array): Initial configuration. If None, uses default IC
        Returns:
            None
        """
        super().__init__(max_timestep, h, k, κ, S, N, M, type, IC)
        self.__S = S
        
        if sink_fnc is None: sink_fnc = lambda r, z: np.zeros((self._N, self._M))
        assert sink_fnc.shape == (self._N, self._M)
        self.__sink_fnc = sink_fnc
        self.__sinked = np.zeros((max_timestep, N, M))

    
    def solve(self) -> None:
        """
        Solves the system for all timesteps.

        Args:
            None
        Returns:
            None
        """
        for t in range(1, self._max_timestep):
            removed_in_sink = self.__sink_fnc*self._solution[t-1]
            self.__sinked[t] = removed_in_sink
            prev_step = self._solution[t-1] - removed_in_sink
            self._solution[t] = ( self._A @ prev_step.flatten() ).reshape((self._N, self._M))


        self._solved = True
    

    def plot_sinked(self, ax: plt.Axes = None) -> None:
        """
        Plots the total sinked concentration over time.

        Args:
            ax (Matplotlib Axes)
        Returns:
            None
        """
        if ax is None: ax = plt.gca()
        ax.plot(self._trange, self.__sinked.sum(axis=(1, 2))/self.__S)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Absorbed in sink")
        ax.grid()


    @property
    def fraction_absorbed(self) -> float:
        return np.sum(self.__sinked)/self.__S