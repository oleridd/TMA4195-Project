import numpy as np
import matplotlib.pyplot as plt
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


def plot_with_silder(arr: np.ndarray, xlabel: str = "$x$", ylabel: str = "$y$", log: bool = False) -> Slider:
    """
    Given a 3D numpy array, plots arr along the 0th axis on a 3D
    plot with xrange and yrange.

    Args:
        arr (TxNxM array): The object to plot, T is the amount of timesteps
        xlabel   (string)
        ylabel   (string)
        log        (bool): Whether to  apply log to plot
    Returns:
        None
    """
    # Initializing plot:
    if log: arr = np.log10(1 + arr)
    ax = plt.figure().add_subplot()
    divider = make_axes_locatable(ax)
    im = ax.imshow(arr[0])
    colorbar_ax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, ax=colorbar_ax, format=LogFormatterMathtext())


    # Adding slider:
    slider_ax = divider.append_axes('top', size='5%', pad=0.05)
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