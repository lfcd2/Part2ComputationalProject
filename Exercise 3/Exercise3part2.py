import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


k1 = 1.34
k2 = 1.6e9
k3 = 8e3
k4 = 4e7
k5 = 1


def create_arrays(length, timestep):
    time_array = np.arange(0, length, timestep)
    X = np.zeros(np.shape(time_array))
    Y = np.zeros(np.shape(time_array))
    Z = np.zeros(np.shape(time_array))
    A = np.zeros(np.shape(time_array))
    B = np.zeros(np.shape(time_array))
    X[0] = 10 ** (-9.8)
    Y[0] = 10 ** (-6.52)
    Z[0] = 10 ** (-7.32)
    A[0] = 0.06
    B[0] = 0.06
    return time_array, X, Y, Z, A, B


def compute_step(X, Y, Z, A, B, timestep, i):
    x, y, z, a, b = X[i], Y[i], Z[i], A[i], B[i]  # 0.06, 0.06  # Concentration of A and B is much greater than X,Y,Z so steady state

    X[i+1] = x + timestep * (k1*a*y - k2*x*y + k3*b*x - 2*k4*x*x)
    Y[i+1] = y + timestep * (-k1*a*y - k2*x*y + k5*z)
    Z[i+1] = z + timestep * (k3*b*x - k5*z)
    # A[i+1] = a + timestep * (-k1*a*y)
    # B[i+1] = b + timestep * (-k3*b*x)

    return X, Y, Z, A, B


def plot_graph(time, X, Y, Z, length):
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(time, X, color='blue', label='X')
    ax.plot(time, Y, color='green', label='Y')
    ax.plot(time, Z, color='red', label='Z')
    ax.legend()
    plt.title('Plot of Concentration against time for the Oregonator reaction')
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Concentration / M')
    ax.set_xlim(0, length)
    plt.show()


def run():
    length = 90
    timestep = 1e-6
    time, X, Y, Z, A, B = create_arrays(length, timestep)
    for i in tqdm(range(int(length/timestep)-1)):
        X, Y, Z, A, B = compute_step(X, Y, Z, A, B, timestep, i)
    plot_graph(time, X, Y, Z, length)


if __name__ == '__main__':
    run()




