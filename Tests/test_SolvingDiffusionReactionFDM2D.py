import matplotlib.pyplot as plt

from Numeric.SolvingDiffusionReactionFDM2D import SolvingDiffusionReactionFDM2D


def test_SolvingDiffusionReactionFDM2D() -> SolvingDiffusionReactionFDM2D:
    """
    Tests the SolvingDiffusionReactionFDM2D class and returns
    the class of sliders to conserve plots.
    """
    system = SolvingDiffusionReactionFDM2D()
    system.solve()
    plt.figure(1)
    system.plot_norm_development()
    plt.figure(2)
    system.plot()
    return system
