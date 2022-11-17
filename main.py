import matplotlib.pyplot as plt
from Tests.test_DiffusionFDM2D import test_DiffusionFDM2D, test_DiffusionFDM2DWithSink
from Numeric.plot_simple_diffusion_solution import plot_simple_diffusion_solution


def main() -> None:
    system = test_DiffusionFDM2DWithSink(500, 0.01, 0.001, M=50, N=50, type="backward")
    plt.show()


if __name__ == "__main__":
    main()