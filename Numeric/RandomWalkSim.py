from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sci

from Utility import minmax


class RandomWalk:

    """ Uniform random walks in infinite space and arbitrary dimensions """

    def __init__(self, N: int, Nstep: int, step: float, D: int = 3) -> None:
        """
        Args:
            N     (int):   Amount of particles to walk
            Nstep (int):   Amount of steps in total.
            step  (float): Step length. Can be input as a suitable Scipy distribution.
            D     (int):   Amount of dimensions. Defaults to 3D.
        """
        self._N = N
        self._Nstep = Nstep
        self.__D = D

        self.__pos = self.__random_walk(N, Nstep, step, D)

    
    def __random_walk(self, N: int, Nstep: int, step: float, D: int) -> np.ndarray:
        """
        Performs a random walk with uniform distribution in
        all directions.

        Args:
            N     (int)
            Nstep (int)
            step  (float)
            D     (int)
        Returns:
            Tensor with each particles position at each timestep
        """
        if   ( hasattr(step, 'dist')          ): step = step.rvs(size=N)
        elif ( isinstance(step, (int, float)) ): step = np.ones(step)
        else: raise ValueError("Type of \"step\" is unrecognized")
        
        # pos: (t x N x D), where t is time
        pos = np.random.uniform(-1, 1, size=(Nstep, N, D))             # Unscaled contribution at each timestep
        pos *= (step[None, :]/np.linalg.norm(pos, axis=2))[:, :, None] # Scaling contributions by step/||vector||. Broadcasting on last index.
        pos = np.cumsum(pos, axis=0)                                   # Summing contributions over time
        return pos

    
    def plot2D(self, timestep = "last") -> None:
        """
        Plots each dimension pair in a subplot.

        Args:
            timestep (int or str): Timestep to plot. Defaults to last.
        Returns:
            None
        """
        assert self.__D >= 2 # Must be at least 2D

        if ( timestep == "last" ):
            timestep = -1

        fig, ax = plt.subplots(self.__D-1, self.__D-1, figsize=(12, 12))

        for i in range(self.__D-1):
            for j in range(i+1, self.__D):
                cax = ax[i, j-i-1] if self.__D > 2 else ax
                cax.plot(
                    self.__pos[timestep, :, i],
                    self.__pos[timestep, :, j],
                    'o'
                    )
                cax.grid()
                cax.set_xlim(minmax(1.1*self.__pos[-1, :, i])) # Always setting scale to that of the last timestep
                cax.set_ylim(minmax(1.1*self.__pos[-1, :, j])) # This makes it easier to compare plots from different timesteps
                cax.set_title(f"$x_{i+1}$-$x_{j+1}$-plot")
                cax.set_xlabel(f"$x_{i+1}$")
                cax.set_ylabel(f"$x_{j+1}$")
    
    @property
    def pos(self) -> np.ndarray:
        return self.__pos