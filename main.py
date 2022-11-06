import scipy.stats as scistat
import matplotlib.pyplot as plt
from Numeric.NeuroRandomWalk import NeuroRandomWalk


def main() -> None:
    step = scistat.lognorm(0.1, 0)
    NRW = NeuroRandomWalk(N=400, Nstep=1000, step=step, Nlayers=20, layer_distr={"type": "simple", "params": 0.4})
    print(NRW)
    NRW.scatter()
    plt.show()


if __name__ == "__main__":
    main()