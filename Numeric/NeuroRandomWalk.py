import numpy as np
import matplotlib.pyplot as plt

from RandomWalkSim import RandomWalk


class NeuroRandomWalk(RandomWalk):
    
    def __init__(self, N: int, Nstep: int, step: float, Nlayers: int, layer_distr: dict, h: float = 15, r: float = 15.22) -> None:
        """
        Args:
            N            (int): Amount of particles to walk
            Nstep        (int): Amount of steps in total.
            step       (float): Step length. Can be input as a suitable Scipy distribution.
            Nlayers      (int): Amount of horizontal layers
            layer_distr (dict): Dictionary on the form {type: (str), params: (tuple or int)}, defining layer transition distribution.
            h          (float): The height of the synaptic cleft. Typically around 15 nm.
        """
        super().__init__(N, Nstep, step, D=2)

        # Handling layer transition distribution:
        if ( layer_distr["type"] == "simple" ):
            self.__distr = lambda l: self.__simple_layer_distribution(l, layer_distr["params"])
        else:
            raise NotImplementedError("Distribution %s is not implemented".format(layer_distr["type"]))

        # Class members:
        self.__Nlayers = Nlayers
        self.__layers_range = np.arange(Nlayers) # Range of possible layers
        self.__layers = self.__simulate_layer_transmission()
        self.__add_layer_positions(h)
        self.__h = h
        self.__r = r


    def __str__(self) -> str:
        return f"Amount at bottom of cylinder: {self.__get_fraction_on_bottom_of_cylinder()}"

    
    def __simulate_layer_transmission(self) -> np.ndarray:
        """
        Performs simulation of layer transmission for each timestep.

        Args:
            None
        Returns:
            layers (2D array (t x n)): Array of layer state for each timestep and each particle
        """
        layers = np.ones((self._Nstep, self._N), dtype=int)*(self.__Nlayers-1)
        for t in range(1, self._Nstep): # Time
            for n in range(self._N):    # Particles
                layers[t, n] = np.random.choice(
                    self.__layers_range,
                    p=self.__distr(layers[t-1, n])
                )
    
        return layers
    

    def __add_layer_positions(self, h: float) -> None:
        """
        Adds positions in z-direction yielded from layers.
        """
        dist_between_layers = h/self.__Nlayers
        z = self.__layers*dist_between_layers
        self._pos = np.dstack((self._pos, z))
        self._D += 1


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
        else:
            distr[l-1], distr[l+1] = p, p
        
        return distr
    

    def __get_fraction_on_bottom_of_cylinder(self) -> float:
        """
        Returns the fraction of particles that are eligable for
        reaction, i.e. that are within the synaptic cleft and
        on the bottom layer.
        """
        s = (self.__layers[-1] == 0)*(np.linalg.norm(self._pos[-1, :, :2], axis=1) <= self.__r)
        return np.sum(s)/self._N

    
    def scatter(self, timestep: str = "last") -> None:
        """
        Generates a scatterplot in 2D or 3D. Requires that pos has
        either 2 or 3 dimensions.
        
        Args:
            timestep (int or str): Timestep to plot. Defaults to last.
        Returns:
            None
        """
        assert self._D in (2, 3)

        if ( timestep == "last" ):
            timestep = -1

        # Scatterplot
        ax = plt.figure().add_subplot(projection="2d" if self._D == 2 else "3d")
        ax.scatter(*self._pos[timestep].T)

        
        # Cylinder of radius r
        z = np.linspace(0, self.__h, 50)
        theta = np.linspace(0, 2*np.pi, 50)
        theta_grid, z_grid=np.meshgrid(theta, z)
        x_grid = self.__r*np.cos(theta_grid)
        y_grid = self.__r*np.sin(theta_grid)

        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, color='red')

    @property
    def layers(self) -> np.ndarray:
        return self.__layers