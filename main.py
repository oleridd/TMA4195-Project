import matplotlib.pyplot as plt
from Numeric.plot_simple_diffusion_solution import plot_simple_diffusion_solution


def main() -> None:
    plot_simple_diffusion_solution(3, N=100, M=300, P=250)
    plt.show()


if __name__ == "__main__":
    main()