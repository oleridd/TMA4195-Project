import numpy as np

from RandomWalkSim import RandomWalk

class NeuroRandomWalk(RandomWalk):
    
    def __init__(self, N: int, Nstep: int, step: float, Nlayers: int, layer_distr: dict) -> None:
        """
        Args:
            N            (int): Amount of particles to walk
            Nstep        (int): Amount of steps in total.
            step       (float): Step length. Can be input as a suitable Scipy distribution.
            Nlayers      (int): Amount of horizontal layers
            layer_distr (dict): Dictionary on the form {type: (str), params: (tuple)}, defining layer transition distribution.
        """
        super().__init__(N, Nstep, step, D=2)
        self.__Nlayers = Nlayers
        self.__layers = np.zeros(Nlayers) # Denotes the layer of each particle
    

    def __simple_layer_distribution(self, l: int, p: float) -> np.ndarray:
        """
        Simple distribution for layer transition in which there is a constant
        probability p of moving layer.
        
        Args:
            l   (int): 
            p (float): Probability of transitioning to layer above OR below
        Returns:
            Discrete distribution of size Nlayers (numpy array)
        """
        distr = np.zeros(self.__Nlayers)
        distr[l] = 1 - 2*p
        if ( l in (0, self.__Nlayers-1) ):
            distr[l] += p
            if   ( l == 0 ): distr[l+1] = p
            elif ( l == self.__Nlayers-1 ): distr[l-1] = p
        
        return distr