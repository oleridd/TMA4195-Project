import matplotlib.pyplot as plt
import numpy as np
from Tests.test_DiffusionFDM2D import test_DiffusionFDM2D, test_DiffusionFDM2DWithSink
from Tests.test_Comparison import comparison
from Tests.test_SolvingDiffusionReactionFDM2D import test_SolvingDiffusionReactionFDM2D
from Numeric.simple_diffusion_solution import plot_simple_diffusion_solution
from Numeric.FDM.SolvingDiffusionReactionFDM2D import SolvingDiffusionReactionFDM2D


def main() -> None:
    max_timestep = 10
    N, M = 50, 50
    h=15e-9/N
    k = 1e-11/max_timestep
    S = 1.057e-13 # mol/l
    system = test_DiffusionFDM2D(timestep=max_timestep, h=h, k=k, M=M, N=N, S=S, type="backward")
    # plot_simple_diffusion_solution(t=1e-9)
    # sliders = comparison(100, h, k, N=40, M=40, S=S)
    # system = test_SolvingDiffusionReactionFDM2D()
    plt.show()


if __name__ == "__main__":
    main()