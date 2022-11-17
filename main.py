import matplotlib.pyplot as plt
from Tests.test_DiffusionFDM2D import test_DiffusionFDM2D, test_DiffusionFDM2DWithSink
from Numeric.plot_simple_diffusion_solution import plot_simple_diffusion_solution


def main() -> None:
    N, M = 50, 50
    system = test_DiffusionFDM2DWithSink(50, h=15e-9/M, k=1e-9/500, M=M, N=N, type="backward")
    plt.show()


if __name__ == "__main__":
    main()