import numpy as np

def minmax(arr: np.ndarray):
    """
    Returns the min and max values of the given array.

    Args:
        arr (array)
    Returns:
        Tuple of the form (min(arr), max(arr))
    """
    return np.min(arr), np.max(arr)