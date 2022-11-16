import scipy.stats as scistat
import matplotlib.pyplot as plt
from Numeric.plot_simple_diffusion_solution import plot_simple_diffusion_solution


def main() -> None:
    plot_simple_diffusion_solution(1)
    plt.show()


if __name__ == "__main__":
    main()