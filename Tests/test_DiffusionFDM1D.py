from Numeric.FDM.DiffusionReactionFDM1D import DiffusionReactionFDM1D


def test_DiffusionReactionFDM1D(timestep: int, h: float, k: float, N: int, S: float) -> DiffusionReactionFDM1D:
    system = DiffusionReactionFDM1D(timestep, h=h, k=k, S=S, N=N)
    system.solve()
    system.plot()
    return system