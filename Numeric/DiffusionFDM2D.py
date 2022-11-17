import numpy as np
import matplotlib.pyplot as plt
from Numeric.Utility import plot_with_silder


class DiffusionFDM2D:

    """
    Finite difference method for the diffusion equation in two dimensions.
    """

    def __init__(self, max_timestep: int, h: float, k: float, S: float, N: int = 10, M: int = 30, type: str = "backward", IC: np.ndarray = None) -> None:
        """
        Args:
            max_timestep (int): The maximum timestep
            h          (float): Spatial step in both z- and r direction
            k          (float): Timestep
            S          (float): Conserved integral over c (the solution)
            N            (int): Amount of spatial points along the z-axis
            M            (int): Amount of spatial points along the r-axis
            type      (string): Whether the method uses forward or backward FDM
            IC     (NxM Array): Initial configuration. If None, uses default IC
        Returns:
            None
        """
        self.__max_timestep = max_timestep
        self.__h = h
        self.__k = k
        self.__N = N
        self.__M = M
        self.__P = M*N # Total amount of gridpoints
        assert type in ("forward", "backward")

        self.__A = self.__construct_matrix(type)
        self.__solution = np.zeros((max_timestep, N, M))
        self.__solution[0] = self.__get_IC(S) if IC is None else IC
        self.__solved = False # True if solve() has been called

    
    def __get_IC(self, S: float) -> np.ndarray:
        """
        Calculates the initial condition in which only the first two gridpoints in
        z-direction and a about 25% of the first gridpoints in r-direction are assigned
        a constant value.

        Args:
            S (float): Conserved integral over c (the solution)

        Returns:
            Initial condition (NxM Array)
        """
        m_r0 = int(0.25*self.__M)
        n_ε  = max(int(0.01*self.__N), 1)
        IC = np.zeros((self.__N, self.__M))
        IC[:n_ε, :m_r0] = S / (self.__h**2*m_r0*n_ε)
        return IC


    def __construct_submatrix(self, boundary: bool = False) -> np.ndarray:
        """
        Constructs NxN tridiagonbal submatrices used on the diagonal of the main
        matrix for the solution.
        
        Args:
            boundary (bool): Whether or not the coresponding row is at the boundary,
                             in which case the diagonal is subtracted 1.
        Returns:
            Tridiagonal submatrix on the form.
            Example when N=4, bounadry=False:
            -3  1  0  0
             1 -4  1  0
             0  1 -4  1
             0  0  1 -3
        """
        offdiag   =                    -np.ones(self.__N-1)
        diag      = (4 - int(boundary))*np.ones(self.__N  )
        submatrix = np.diag(offdiag, -1) + np.diag(diag, 0) + np.diag(offdiag, 1)
        submatrix[ 0,  0] -= 1
        submatrix[-1, -1] -= 1

        return -submatrix


    def __construct_matrix(self, type: str) -> np.ndarray:
        """
        Constructs the main updating matrix of the flattened system.

        Args:
            type (string): Whether the method uses forward or backward FDM
        Returns:
            System matrix A (constant with time)
        """
        A = np.zeros((self.__P, self.__P))
        r = self.__k / self.__h**2

        # Precalculating matrices to accelerate runtime
        matrix_bndry = self.__construct_submatrix(boundary=True )
        matrix_inner = self.__construct_submatrix(boundary=False)
        get_matrix = lambda bndry: matrix_bndry if bndry else matrix_inner

        # Applying to main matrix
        A[:self.__N, :self.__N] = matrix_bndry # Constructing the upperleft submatrix
        for i in range(2, self.__M+1):
            n0, n1, n2 = (i-2)*self.__N, (i-1)*self.__N, i*self.__N
            A[n1:n2, n1:n2] = get_matrix(i==self.__M)
            A[n1:n2, n0:n1] = np.identity(self.__N)
            A[n0:n1, n1:n2] = np.identity(self.__N)
        
        # Returning the correct type of matrix
        if   type == "forward":
            return np.identity(self.__P) + r*A
        elif type == "backward":
            return np.linalg.solve(np.identity(self.__P) - r*A, np.identity(self.__P))

    
    def solve(self) -> None:
        """
        Solves the system for all timesteps.

        Args:
            None
        Returns:
            None
        """
        for t in range(1, self.__max_timestep):
            self.__solution[t] = ( self.__A @ self.__solution[t-1].flatten() ).reshape((self.__N, self.__M))

        self.__solved = True
    

    def __assert_solved(self) -> None:
        """
        Raises an error if solve() method has not been called at the time
        of this method being called.
        """
        if not self.__solved:
            raise RuntimeError("DiffusionFDM2D: System has note yet been solve. Please call the \'solve()\' method.")

    
    def plot(self, timestep: int = -1, slider: bool = False) -> None:
        """
        Plots the solution in a 3D surface plot.

        Args:
            Timestep (int): Timestep to plot, defaults to last.
            slider  (bool): Whether or not to plot with slider
        Returns:
            None
        """
        self.__assert_solved()

        rrange = np.arange(0, self.__M*self.__h, self.__h)
        zrange = np.arange(0, self.__N*self.__h, self.__h)


        if not slider:
            R, Z = np.meshgrid(
                rrange,
                zrange
            )
            ax = plt.figure().add_subplot(projection="3d")
            ax.plot_surface(R, Z, self.__solution[timestep])
            ax.set_xlabel("$r$")
            ax.set_ylabel("$z$")
        
        else:
            self.__slider = plot_with_silder(self.__solution, xlabel="$r$", ylabel="$z$", log=True)
    

    @property
    def solution(self) -> np.ndarray:
        self.__assert_solved()
        return self.__solution