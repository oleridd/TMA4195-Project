import numpy as np
import matplotlib.pyplot as plt

from Numeric.Utility import plot_with_slider_2D


class DiffusionReactionFDM1D:

    def __init__(self, max_timestep: int, h: float, k: float, κ: float = 8e-7, S: float = 500, N: int = 10, IC_N: np.ndarray = None, IC_R: np.ndarray = None) -> None:
        """
        Args:
            max_timestep (int): The maximum timestep
            h          (float): Spatial step in both z- and r direction
            k          (float): Timestep
            κ          (float): Diffusion coefficient [m^2/s]
            S          (float): Conserved integral over c (the solution)
            N            (int): Amount of spatial points along the z-axis
            IC_N     (N Array): Initial configuration for neurotransmitters. If None, uses default IC
            IC_R     (N Array): Initial configuration for neurotransmitters. If None, uses default IC
        Returns:
            None
        """
        self._max_timestep = max_timestep
        self._h = h
        self._k = k
        self._N = N
        self._r = κ*k/h**2

        # self._k1 = 100*(4e6)/((6.022e23)*(220e-9**2*np.pi*self._N*self._h*1000))
        self._k1 = 1e20
        self._k2 = 0

        # Neurotransmitter concentration
        self._A = self._construct_diffusion_matrix()
        self._N_solution = np.zeros((max_timestep, N))
        self._N_solution[0] = self._get_IC(S, location="top") if IC_N is None else IC_N
        
        # Receptor concentration:
        self._R_solution = np.zeros((max_timestep, N))
        self._R_solution[0] = self._get_IC(S, location="bottom") if IC_R is None else IC_R

        # Bound receptor concentration:
        self._Rb_solution = np.zeros((max_timestep, N))
        self._Rb_solution[0] = np.zeros(self._N)
        

        self._solved = False # True if solve() has been called
    
    
    def _construct_diffusion_matrix(self) -> np.ndarray:
        """
        Constructs the system matrix for 1D diffusion.
        Example for N = 4:
        -1  1  0  0
         1 -2  1  0
         0  1 -2  1
         0  0  1 -1

        Args:
            None
        Returns:
            None
        """
        offdiag = np.ones(self._N-1)
        diag    = -2*np.ones(self._N)
        diag[0], diag[-1] = -1, -1
        return np.diag(offdiag, -1) + np.diag(diag) + np.diag(offdiag, +1)
    

    def _get_IC(self, S: float, location: str = "top") -> np.ndarray:
        """
        Generates standard initial condition matrix for the 1D case,
        in which particles are distributed over 5% of the are on the
        top of the synaptic cleft.

        Args:
            S         (float): Density of neurotransmitters
            location (string): Either top or bottom
        Returns:
            1D IC
        """
        IC = np.zeros(self._N)
        n_ε = max(int(0.05*self._N), 1)
        if location == "top":      IC[:n_ε ] = S
        elif location == "bottom": IC[-n_ε:] = S
        return IC
    
    
    def solve(self) -> None:
        """
        Solves the system and stores the solution as class variables.

        Args:
            None
        Returns:
            None
        """
        Nsol = self._N_solution
        Rsol = self._R_solution
        Rbsol = self._Rb_solution

        for t in range(1, self._max_timestep):

            Rsol[t]  = Rsol[t-1]  - self._k*self._k1*Rsol[t-1]*Nsol[t-1] + self._k*self._k2*Rbsol[t-1]
            Rbsol[t] = Rbsol[t-1] + self._k*self._k1*Rsol[t-1]*Nsol[t-1] - self._k*self._k2*Rbsol[t-1]
            
            Nsol[t] = np.linalg.solve(
                np.identity(self._N) - self._r*self._A + self._k*self._k1*np.diag(Rsol[t]),
                Nsol[t-1] + self._k*self._k2*Rbsol[t]
            )

        self._N_solution = Nsol
        self._R_solution = Rsol
        self._Rb_solution = Rbsol

        self._solved = True

    
    def _assert_solved(self) -> None:
        """
        Raises an error if solve() method has not been called at the time
        of this method being called.
        """
        if not self._solved:
            raise RuntimeError("DiffusionFDM2D: System has note yet been solve. Please call the \'solve()\' method.")
    
        
    def plot(self) -> None:
        """
        Plots the solution.
        """
        self.__sliders = plot_with_slider_2D(
            [self._N_solution, self._R_solution, self._Rb_solution],
            ["t"],
            ["$c_N$", "$c_R$", "$c_{Rb}$"],
            axis_to_plot=1,
            xrange=np.linspace(0, self._N*self._h, self._N),
            xlabel="z",
            ylabel="concentration"
        )