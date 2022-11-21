import matplotlib.pyplot as plt
from Tests.test_DiffusionFDM2D import test_DiffusionFDM2D, test_DiffusionFDM2DWithSink
from Tests.test_Comparison import comparison
from Numeric.simple_diffusion_solution import plot_simple_diffusion_solution


def main() -> None:
    N, M = 40, 40
    h = h=15e-9/M
    k = 1e-9/500
    # system = test_DiffusionFDM2D(50, h=15e-9/M, k=1e-9/500, M=M, N=N, type="backward")
    # plot_simple_diffusion_solution(1, N=25, M=60)
    sliders = comparison(100, h, k, N=10, M=10)
    plt.show()


if __name__ == "__main__":
    main()