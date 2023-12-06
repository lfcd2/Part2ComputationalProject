import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Rate constants (global)
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
    X[0] = 10 ** (-9.8)
    Y[0] = 10 ** (-6.52)
    Z[0] = 10 ** (-7.32)
    return time_array, X, Y, Z


def compute_step(x_array, y_array, z_array, timestep, i):
    x, y, z = x_array[i], y_array[i], z_array[i]
    a, b = 0.06, 0.06  # Concentration of A and B is much greater than X,Y,Z so steady state

    x_array[i + 1] = x + timestep * (k1 * a * y - k2 * x * y + k3 * b * x - 2 * k4 * x * x)
    y_array[i + 1] = y + timestep * (-k1 * a * y - k2 * x * y + k5 * z)
    z_array[i + 1] = z + timestep * (k3 * b * x - k5 * z)


def plot_graph(time, x_array, y_array, z_array, length):
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(time, x_array, color='blue', label='X')
    ax.plot(time, y_array, color='green', label='Y')
    ax.plot(time, z_array, color='red', label='Z')
    ax.legend()
    plt.title('Plot of Concentration against time for the Oregonator reaction')
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Concentration / M')
    ax.set_xlim(0, length)
    plt.show()


def run():
    length = 1
    timestep = 1e-6
    time, X, Y, Z = create_arrays(length, timestep)
    for i in tqdm(range(int(length/timestep)-1)):
        compute_step(X, Y, Z, timestep, i)
    plot_graph(time, X, Y, Z, length)


if __name__ == '__main__':
    run()




