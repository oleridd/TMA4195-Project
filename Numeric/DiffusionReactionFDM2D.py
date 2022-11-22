import numpy as np

from Numeric.DiffusionFDM2D import DiffusionFDM2D


class DiffusionReactionFDM2D(DiffusionFDM2D):

    def __init__(self, max_timestep: int, h: float, k: float, κ: float = 8e-7, r: float = 0, S: float = 500, N: int = 10, M: int = 30, type: str = "backward", IC: np.ndarray = None, has_diffusion: bool = True):
        """
        Args:
            max_timestep (int): The maximum timestep
            h            (float): Spatial step in both z- and r direction
            k            (float): Timestep
            κ            (float): Diffusion coefficient [m^2/s]
            S            (float): Conserved integral over c (the solution)
            N              (int): Amount of spatial points along the z-axis
            M              (int): Amount of spatial points along the r-axis
            type        (string): Whether the method uses forward or backward FDM
            IC       (NxM Array): Initial configuration. If None, uses default IC
            has_diffusion (bool): Whether or not the solved particle has diffusion (C_N)
        Returns:
            None
        """
        self.__has_diffusion = has_diffusion
        super().__init__(max_timestep, h, k, κ, S, N, M, type, IC)
      
    
    def _construct_matrix(self, type: str) -> np.ndarray:
        return super()._construct_system_matrix() if self.__has_diffusion else None
        
    
    def set_solution(self, t: int, sol: np.ndarray) -> None:
        """
        Sets the solution at timestep t.

        Args:
            t         (int): Timestep at which to set the solution
            sol (NxM array): Solution to set
        Returns:
            None
        """
        self._solution[t] = sol
    

    def get_solution(self, t: int) -> np.ndarray:
        """
        Gets the solution at timestep t.

        Args:
            t (int): Timestep at which to get the solution
        Returns:
            Solution at time t (NxM array)
        """
        return self._solution[t]


    @property
    def A(self) -> np.ndarray:
        return self._A