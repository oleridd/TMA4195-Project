import numpy as np

from Numeric.DiffusionFDM2D import DiffusionFDM2D


class DiffusionReactionFDM2D(DiffusionFDM2D):

    def __init__(self, max_timestep: int, h: float, k: float, κ: float = 8e-7, S: float = 500, N: int = 10, M: int = 30, type: str = "backward", IC: np.ndarray = None, has_diffusion: bool = True):
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
        super().__init__(max_timestep, h, k, κ, S, N, M, type, IC)
        self.__has_diffusion = has_diffusion
      
    
    def _construct_matrix(self, type: str) -> np.ndarray:
        if self.__has_diffusion:
            A = super()._construct_matrix(type)
        else:
            A = np.identity(self.__P)
    