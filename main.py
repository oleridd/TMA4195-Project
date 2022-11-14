import scipy.stats as scistat
import matplotlib.pyplot as plt
from Numeric.NeuroRandomWalk import NeuroRandomWalk


def main() -> None:
    step = scistat.lognorm(.1, -.5)
    NRW = NeuroRandomWalk(N=400, Nstep=5000, step=step, Nlayers=20, layer_distr={"type": "simple", "params": 0.4})
    print(NRW)
    NRW.scatter()
    plt.show()


if __name__ == "__main__":
    main()