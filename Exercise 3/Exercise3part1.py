import numpy as np
import matplotlib.pyplot as plt


def run_ex3p1() -> None:
    """
    Computes the graph of the protein folding dependence on Urea conc.
    :return: None
    """
    urea = np.arange(0, 8, 0.01)

    kf15 = 26000*np.exp(-1.68*urea)
    ku15 = 0.06*np.exp(0.95*urea)
    kf16 = 730*np.exp(-1.72*urea)
    ku16 = 7.5e-4*np.exp(1.2*urea)

    concentration_I = 1/((ku15/kf15)+(kf16/ku16)+1)
    concentration_D = concentration_I * ku15/kf15
    concentration_N = concentration_I * kf16/ku16

    plt.plot(urea, concentration_D, color='blue', label='D')
    plt.plot(urea, concentration_I, color='green', label='I')
    plt.plot(urea, concentration_N, color='red', label='N')
    plt.legend(loc='center left')
    plt.xlabel('Fraction of species')
    plt.ylabel('[Urea]/M')
    plt.title('Equilibrium fractions of protein states against urea concentration')
    plt.xlim(0, 8)
    plt.ylim(-0.01, 1.01)
    plt.savefig('Part1Output.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    run_ex3p1()
