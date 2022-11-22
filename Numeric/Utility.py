import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LogFormatterMathtext


def minmax(arr: np.ndarray):
    """
    Returns the min and max values of the given array.

    Args:
        arr (array)
    Returns:
        Tuple of the form (min(arr), max(arr))
    """
    return np.min(arr), np.max(arr)


def get_absorption_frac(sim, R: float, ε: float) -> float:
    """
    Given a RandomWalkSingleBoundary simulation, studies the pos variable
    to find the fraction of absorbed particles (particles that have reached)
    the upper boundary and are within a certain radius).

    Args:
        sim (RandomWalkSingleBoundary): RandomWalk simulation with boundary as defined in RandomWalkSingleBoundary.py
        R                      (float): Radius at boundary required for absorption
        ε                      (float): Acceptable distance from boundary to be absorbed (in z-direction)
    Returns:
        Fraction of absorbed particles
    """
    pos = sim.pos
    conditional = ( sim.z_boundary - pos[:, :, -1] < ε ) * ( pos[:, :, 0]**2 + pos[:, :, 1]**2 < R**2 )
    return np.sum(np.any(conditional, axis=0))/pos.shape[1] # If at least one timestep is true, set to true


def plot_with_silder(arr: np.ndarray, xlabel: str = "$x$", ylabel: str = "$y$", log: bool = False, ax: plt.Axes = None, cb: bool = False) -> Slider:
    """
    Given a 3D numpy array, plots arr along the 0th axis on a 3D
    plot with xrange and yrange.

    Args:
        arr (TxNxM array): The object to plot, T is the amount of timesteps
        xlabel      (string)
        ylabel      (string)
        log           (bool): Whether to  apply log to plot
        ax (Matplotlib Axes)
        cb            (bool): Whether to include a colorbar
    Returns:
        None
    """
    # Initializing plot:
    if log: arr = np.log10(1 + arr)
    if ax is None: ax = plt.gca()

    divider = make_axes_locatable(ax)
    im = ax.imshow(arr[0])
    if cb:
        colorbar_ax = divider.append_axes('left', size='5%', pad=0.15)
        plt.colorbar(im, ax=colorbar_ax)


    # Adding slider:
    slider_ax = divider.append_axes('top', size='5%', pad=0.15)
    slider = Slider(
        ax=slider_ax,
        label="Timestep",
        valmin=0,
        valmax=len(arr)-1,
        valinit=0,
        valstep=np.arange(len(arr))
    )


    # Updating plot from slider:
    def update(val) -> None:
        im.set_data(arr[int(val)])
        plt.draw()
    
    slider.on_changed(update)


    # Returning to retain slider functionality:
    return slider


def plot_with_slider_2D(arrs: list, slider_labels: list, plot_labels: list, axis_to_plot: int, xrange: np.ndarray, ax: plt.Axes = None, xlabel: str = "", ylabel: str = "", **plot_kwargs) -> None:
    """
    Given a list of ND arrays, plots each array along a given axis, with sliders to
    adjust the additional axes.

    Args:
        arrs    (list(ND Arrays)): List of arrays to plot in the same plot
        slider_labels (list[str]): Labels for the sliders
        plot_labels   (list[str]): Labels (legends) for the plots
        axis_to_plot        (int): The axis to plot on the x-axis
        xrange         (1D array): Range of the x-axis in the plot
        ax      (Matplotlib Axes)
        xlabel           (string)
        ylabel           (string)
    Returns:
        None
    """
    # Assertions
    axes = arrs[0].shape # Axes for all arrays
    assert np.all([(arr.shape == axes) for arr in arrs])
    assert len(axes) == len(slider_labels)+1
    assert len(arrs) == len(plot_labels)
    assert axes[axis_to_plot] == len(xrange)

    
    # Useful variables in function scope:
    atp = axis_to_plot                                 # The axis to be plotted
    current_indices = np.zeros(len(axes)-1, dtype=int) # The indices that are plotted (excluding axis_to_plot) at any time


    # Initializing plots:
    ax = plt.gca() if ax is None else ax
    lines = [ax.plot(
        xrange,
        get_line_along_axis(arr, axis=atp, indices=current_indices),
        label=lab,
        **plot_kwargs
        )[0] for arr, lab in zip(arrs, plot_labels)]
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


    # Adding sliders:
    sliders = []
    divider = make_axes_locatable(ax) # Used to add slider axes
    for i, lab in zip(range(1, len(axes)), slider_labels):
        slider_ax = divider.append_axes('top', size='5%', pad=0.1)
        sliders.append(
            Slider(
                ax=slider_ax,
                label=lab,
                valmin=0,
                valmax=axes[i],
                valinit=0,
                valstep=np.arange(axes[i])
            )
        )


    # Updating plot from sliders:
    for i in range(1, len(axes)):
        def update(val, i=i):
            for k, line in enumerate(lines):
                current_indices[i-1] = int(val)
                line.set_ydata(get_line_along_axis(arrs[k], axis=atp, indices=current_indices))
        sliders[i-1].on_changed(update)


    return sliders


def get_line_along_axis(arr: np.ndarray, axis: int, indices: tuple) -> np.ndarray:
    """
    Given some ND array, an axis and a tuple of coordinates, retrieves
    the line (1D array) along the given axis at the specific coordinates.

    Args:
        arr  (ND array): Array to retrieve from
        axis      (int): The axis along which to retrieve a vector
        indices (tuple): Indices denoting the line to retrieve
    Returns:
        1D array of the vector located at the specified line
    """
    slc = list(indices)
    slc.insert(axis, slice(None))
    return arr[tuple(slc)]


class DiffusionReaction2DStdConfig:

    """
    Standard config for SolvingDiffusionReaction class
    """

    def __init__(self, particle_type: str) -> None:
        """
        Args:
            particle_type (str): Either N, R or Rb
        """
        self.__max_timestep = 25
        self.__N = 25
        self.__M = 500
        self.__h = 15e-9/self.__N
        self.__k = 1e-9/self.__max_timestep
        self.__κ = 8e-7
        self.__r = 220e-9
        self.__particle_type = particle_type
        if   particle_type == "N": self.__S = 5000/(self.__r**2*np.pi*self.__h)
        elif particle_type == "R": self.__S = (1000)/(np.pi*(self.__r*1e6)**2)
        elif particle_type == "Rb": self.__S = 0
        else: raise RuntimeError("DiffusionReaction2DStdConfig: Invalid particle type")

        self.__IC = self.__construct_IC()

        self.__config = self.__construct_config()


    def __construct_config(self):
        """
        Constructs config dict.

        Args:
            None
        Returns:
            None
        """
        return {
            'max_timestep': self.__max_timestep,
            'N': self.__N,
            'M': self.__M,
            'h': self.__h,
            'k': self.__k,
            'κ': self.__κ,
            'r': self.__r,
            'S': self.__S,
            'IC': self.__IC
        }


    def get_basic_parameters(self) -> tuple:
        """
        Returns the basic parameters of the config:
        - N and M
        - h and k

        Args:
            None
        Returns:
            None
        """
        return self.__max_timestep, self.__N, self.__M, self.__h, self.__k

    
    def __construct_IC(self) -> np.ndarray:
        """
        Generates IC depending on the particle type.

        Args:
            None
        Returns:
            IC (NxM array)
        """
        IC = np.zeros((self.__N, self.__M))

        if self.__particle_type in ('N', 'R'):
            m_r0 = int(self.__r/self.__h)
            n_ε  = max(int(0.25*self.__N), 1)
            if self.__particle_type == 'N':
                IC[:n_ε, :m_r0] = self.__S / (n_ε + m_r0)
            elif self.__particle_type == 'R':
                IC[-int(n_ε*4):, :int(m_r0*4)] = self.__S / (n_ε + m_r0)
        
        return IC
    

    def set_parameter(self, parameter: str, val) -> None:
        """
        Sets the given parameters.
        """
        self.__config[parameter] = val

        # Updating dependent variables:
        self.__h = 15e-9/self.__config['N']
        self.__k = 1e-9/self.__config['max_timestep']
        if   self.__particle_type == "N": self.__S = 5000/(self.__config['r']**2*np.pi*self.__config['h'])
        elif self.__particle_type == "R": self.__S = (1000e-6)/self.__config['h']
        elif self.__particle_type == "Rb": self.__S = 0
        self.__IC = self.__construct_IC()

        # Updating dependent dictionary
        self.__config = self.__construct_config()

    
    @property
    def config(self) -> dict:
        return self.__config