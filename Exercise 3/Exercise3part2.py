import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
from scipy import signal


class Cell:

    def __init__(self, position, duration, timestep, full=False, number_of_cells=1):
        self.position = position
        self.duration = duration
        self.timestep = timestep
        self.time = np.arange(0, self.duration, self.timestep)
        X = np.zeros(np.shape(self.time))
        Y = np.zeros(np.shape(self.time))
        Z = np.zeros(np.shape(self.time))
        if full:
            X[0] = (10 ** (-9.8)) * number_of_cells
            Y[0] = (10 ** (-6.52)) * number_of_cells
            Z[0] = (10 ** (-7.32)) * number_of_cells
        self.x_array = X
        self.y_array = Y
        self.z_array = Z

    def reactions(self, i):
        k1 = 1.34
        k2 = 1.6e9
        k3 = 8e3
        k4 = 4e7
        k5 = 1
        x, y, z = self.x_array[i], self.y_array[i], self.z_array[i]
        a, b = 0.06, 0.06  # Concentration of A and B is much greater than X,Y,Z so steady state

        self.x_array[i + 1] = x + self.timestep * (k1 * a * y - k2 * x * y + k3 * b * x - 2 * k4 * x * x)
        self.y_array[i + 1] = y + self.timestep * (-k1 * a * y - k2 * x * y + k5 * z)
        self.z_array[i + 1] = z + self.timestep * (k3 * b * x - k5 * z)


def plot_graph(time, x_array, y_array, z_array, cell_number):
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(time, x_array, color='blue', label='X')
    ax.plot(time, y_array, color='green', label='Y')
    ax.plot(time, z_array, color='red', label='Z')
    ax.legend()
    plt.title(f'Plot of Concentration against time for the Oregonator reaction ({cell_number})')
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Concentration / M')
    ax.set_xlim(0, time[-1])
    ax.set_ylim(2e-11, 0.002)


def diffuse(cells, i, timestep):
    s = 0.05 * timestep  # D * dt/dv^2    dt = timestep, dv^2 =1, D = 0.05
    temp_storage = []
    for pos, cell in enumerate(cells):

        if pos == 0:
            pass
            x = s * (cells[1].x_array[i] - cell.x_array[i])
            y = s * (cells[1].y_array[i] - cell.y_array[i])
            z = s * (cells[1].z_array[i] - cell.z_array[i])
        elif pos == len(cells)-1:
            x = s * (cells[-2].x_array[i] - cell.x_array[i])
            y = s * (cells[-2].y_array[i] - cell.y_array[i])
            z = s * (cells[-2].z_array[i] - cell.z_array[i])
        else:
            x = s * (cells[pos-1].x_array[i] - 2*cell.x_array[i] + cells[pos+1].x_array[i])
            y = s * (cells[pos-1].y_array[i] - 2*cell.y_array[i] + cells[pos+1].y_array[i])
            z = s * (cells[pos-1].z_array[i] - 2*cell.z_array[i] + cells[pos+1].z_array[i])
        temp_storage.append((x, y, z))
    for pos, cell in enumerate(cells):
        cell.x_array[i] += temp_storage[pos][0]
        cell.y_array[i] += temp_storage[pos][1]
        cell.z_array[i] += temp_storage[pos][2]


def intensity_plot(cells, timestep):

    xs, ys, zs = [], [], []
    for c in cells:
        xtemp = np.where(c.x_array < 1e-11, 1e-11, c.x_array)
        ytemp = np.where(c.y_array < 1e-11, 1e-11, c.y_array)
        ztemp = np.where(c.z_array < 1e-11, 1e-11, c.z_array)
        while len(xtemp) > 2**23:
            xtemp = signal.resample(xtemp, int(len(xtemp) / 2))
            ytemp = signal.resample(ytemp, int(len(ytemp) / 2))
            ztemp = signal.resample(ztemp, int(len(ztemp) / 2))
        xs.append(xtemp)
        ys.append(ytemp)
        zs.append(ztemp)

    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
    titles = ['X', 'Y', 'Z']
    my_plot = None
    for i, data in enumerate([xs, ys, zs]):
        ax = axs[i]
        my_plot = ax.imshow(data, interpolation='none', aspect='auto', norm=mpl.colors.LogNorm(vmin=1e-11, vmax=1e-3))
        ax.set_xlabel('Time / s')
        ax.set_ylabel('Cell')
        ax.set_title(f'Concentrations of {titles[i]}')
        y_ints = range(len(data))
        ax.set_yticks(y_ints)
        ax.set_ylim(0.5, len(data)-0.5)
        ticks = ax.get_xticks()
        ax.set_xticks(ticks, labels=np.round(ticks*timestep, 2))
        ax.set_xlim(0, len(data[0]))
    fig.colorbar(mappable=my_plot, label='Concentration / M')


def run():
    duration = float(input('Duration of simulation: '))
    timestep = 1e-6
    number_of_cells = int(input('number of cells (type 1 for part A): '))+1
    cell_list = []
    for position in range(number_of_cells):
        full = True if position == 0 else False
        new_cell = Cell(position, duration, timestep, full, number_of_cells)
        cell_list.append(new_cell)

    if number_of_cells == 2:  # PART A
        cell = cell_list[0]
        for i in tqdm(range(int(duration / timestep) - 1)):
            cell.reactions(i)
        plot_graph(cell.time, cell.x_array, cell.y_array, cell.z_array, 0)

    else:  # PART B
        for i in tqdm(range(int(duration/timestep)-1)):
            for cell in cell_list:
                cell.reactions(i)
            diffuse(cell_list, i+1, timestep)
        intensity_plot(cell_list, timestep)
    plt.show()


if __name__ == '__main__':
    run()
