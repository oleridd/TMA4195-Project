import numpy as np


class DiffusionFDM2D:

    """
    Finite difference method for the diffusion equation in two dimensions.
    """

    def __init__(self, max_timestep: int, S: float, N: int = 10, M: int = 30, IC: np.ndarray = None) -> None:
        """
        Args:
            max_timestep (int): The maximum timestep
            S          (float): Conserved integral over c (the solution)
            N            (int): Amount of spatial points along the z-axis
            M            (int): Amount of spatial points along the r-axis
            IC     (NxM Array): Initial configuration. If None, uses default IC.
        Returns:
            None
        """
        self.__max_timestep = max_timestep
        self.__N = N
        self.__M = M
        self.__P = M*N # Total amount of gridpoints

        self.__A = self.__construct_matrix()
        self.__solution = np.zeros((max_timestep, N, M))
        self.__solution[0] = self.__get_IC(S) if IC is None else IC
        self.__timestep = 1   # Keeps track of timestep. Starts at 1 due to I.C.
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
        IC = np.zeros((self.__N, self.__M))
        IC[:2, :m_r0] = S / m_r0
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
            Example when N=4:
             3 -1  0  0
            -1  4 -1  0
             0 -1  4 -1
             0  0 -1  3
        """
        offdiag   =                    -np.ones(self.__N-1)
        diag      = (4 - int(boundary))*np.ones(self.__N  )
        submatrix = np.diag(offdiag, -1) + np.diag(diag, 0) + np.diag(offdiag, 1)
        submatrix[ 0,  0] -= 1
        submatrix[-1, -1] -= 1

        return submatrix


    def __construct_matrix(self) -> np.ndarray:
        """
        Constructs the main updating matrix of the flattened system.

        Args:
            None
        Returns:
            System matrix A (constant with time)
        """
        A = np.zeros((self.__P, self.__P))

        A[0:self.__N, 0:self.__N] = self.__construct_submatrix(boundary=True) # Constructing the upperleft submatrix
        for i in range(2, self.__M+1):
            n0, n1, n2 = (i-2)*self.__N, (i-1)*self.__N, i*self.__N
            A[n1:n2, n1:n2] = self.__construct_submatrix(boundary=(i==self.__M))
            A[n1:n2, n0:n1] = -np.identity(self.__N)
            A[n0:n1, n1:n2] = -np.identity(self.__N)
        
        return A

    
    def solve(self) -> None:
        """
        Solves the system for all timesteps.

        Args:
            None
        Returns:
            None
        """
        for t in range(1, self.__max_timestep):
            self.__solution[t] = ( self.__A@self.__solution[t-1].flatten() ).reshape((self.__N, self.__M))

        self.__solved = True

    
    @property
    def solution(self) -> np.ndarray:
        if self.__solved: return self.__solution
        else: raise RuntimeError("DiffusionFDM2D: System has note yet been solve. Please call the \'solve()\' method.")