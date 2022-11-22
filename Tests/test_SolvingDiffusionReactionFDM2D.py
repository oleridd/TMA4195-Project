from Numeric.SolvingDiffusionReactionFDM2D import SolvingDiffusionReactionFDM2D


def test_SolvingDiffusionReactionFDM2D() -> SolvingDiffusionReactionFDM2D:
    """
    Tests the SolvingDiffusionReactionFDM2D class and returns
    the class of sliders to conserve plots.
    """
    system = SolvingDiffusionReactionFDM2D()
    system.solve()
    system.plot()
    return system