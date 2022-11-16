import numpy as np

from Numeric.RandomWalkSim import RandomWalk


class RandomWalkSingleBoundary(RandomWalk):

    """ Uniform random walks in space bounded by an interval in third dimension """

    def __init__(self, N: int, Nstep: int, step: float, z_boundary: float) -> None:
        """
        Args:
            N            (int): Amount of particles to walk
            Nstep        (int): Amount of steps in total.
            step       (float): Step length in each dimension. Can be input as a suitable Scipy distribution.
            D            (int): Amount of dimensions. Defaults to 3D.
            z_boundary (float): The upper boundary on z.
        """
        self.__z_boundary = z_boundary
        super().__init__(N, Nstep, step, D=3)

    
    def _random_walk(self, N: int, Nstep: int, step: float, D: int) -> np.ndarray:
        """
        Performs a random walk with uniform distribution in all directions.
        Assuming particles start at first boundary in x, y = 0.

        Args:
            N            (int)
            Nstep        (int)
            step       (float)
            D            (int)
            z_boundary (float)
        Returns:
            Tensor with each particles position at each timestep
        """
        if   ( hasattr(step, 'dist')          ): step = step.rvs(size=(Nstep, N, D))
        elif ( isinstance(step, (int, float)) ): step = step*np.ones((Nstep, N, D))
        elif ( isinstance(step, np.ndarray)   ): pass
        else: raise ValueError("Type of \"step\" is unrecognized")
        
        # pos: (t x N x D), where t is time
        pos = np.zeros((Nstep, N, D))
        pos[:, :, :-1] = super()._random_walk(N, Nstep, step[:, :, :-1], D-1) # Calling super method for x and y
        for t in range(1, Nstep):
            new_z = pos[t-1, :, -1] + step[t, :, -1]
            new_z[new_z < 0] = 0
            new_z[new_z > self.__z_boundary] = self.__z_boundary
            pos[t, :, -1] = new_z

        return pos
    
    
    @property
    def z_boundary(self) -> float:
        return self.__z_boundary