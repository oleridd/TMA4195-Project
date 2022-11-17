import matplotlib.pyplot as plt
from Tests.test_DiffusionFDM2D import test_DiffusionFDM2D
from Numeric.plot_simple_diffusion_solution import plot_simple_diffusion_solution


def main() -> None:
    system = test_DiffusionFDM2D(100, 0.01, 0.001, M=100, N=100)
    plt.show()


if __name__ == "__main__":
    main()