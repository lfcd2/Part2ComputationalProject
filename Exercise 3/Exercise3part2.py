import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as anim
# from mpl_toolkits.mplot3d import Axes3D
from scipy import signal


class Cell:

    def __init__(self, position, duration, timestep, full=False, number_of_cells=1):
        self.position = position
        self.duration = duration
        self.timestep = timestep
        self.neighbours = []
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

    def setup_neighbours(self, cell_array, source_pos, grid_size):
        pos = self.position
        # left
        if pos[0] <= source_pos[0] and pos[0] != 0:
            self.neighbours.append(cell_array[pos[0] - 1, pos[1]])
        # right
        if pos[0] >= source_pos[0] and pos[0] != grid_size-1:
            self.neighbours.append(cell_array[pos[0] + 1, pos[1]])
        # up
        if pos[1] <= source_pos[1] and pos[1] != 0:
            self.neighbours.append(cell_array[pos[0], pos[1] - 1])
        # down
        if pos[1] >= source_pos[1] and pos[1] != grid_size-1:
            self.neighbours.append(cell_array[pos[0], pos[1] + 1])

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

    def diffusion(self, i):
        num_of_neighbours = len(self.neighbours)

        #if i == 100000:
         #   print(self.neighbours, self.position)


        cell_x = self.x_array[i + 1]
        delta_x = cell_x * 1e-15
        cell_y = self.y_array[i + 1]
        delta_y = cell_y * 1e-15
        cell_z = self.z_array[i + 1]
        delta_z = cell_z * 1e-15

        self.x_array[i + 1] = cell_x - delta_x
        self.y_array[i + 1] = cell_y - delta_y
        self.z_array[i + 1] = cell_z - delta_z

        for n in self.neighbours:
            n.x_array[i + 1] += delta_x / num_of_neighbours
            n.y_array[i + 1] += delta_y / num_of_neighbours
            n.z_array[i + 1] += delta_z / num_of_neighbours



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


def make_animation(cells, timestep, duration, animation_timestep, gridsize, slow):
    fig, (ax, cbar) = plt.subplots(1, 2, width_ratios=([7, 1]))
    norm = mpl.colors.LogNorm(vmin=1e-18, vmax=1e-3)

    def anim_func(frame):
        i = int(frame * (animation_timestep/timestep))
        data = [[c.z_array[i] for c in cell_row] for cell_row in cells] # np.where(c.x_array[i] < 1e-18, 1e-18, c.x_array[i])
        ax.pcolor(data, norm=norm)

    plt.colorbar(mappable=mpl.cm.ScalarMappable(norm=norm), cax=cbar)

    ax.set_title(f'Concentration of Z. Slowed by a factor of {slow}.')
    cbar.set_ylabel(fr'Concentration / mol dm$^-3$')
    fig.tight_layout()
    animation = anim.FuncAnimation(fig, anim_func, interval=animation_timestep*1000*slow,
                                   frames=(int(duration / animation_timestep))+1)
    animation.save('Part2BOutput.mp4')


def run():
    inp = input('Part A or B:')
    if inp.upper() == 'A':
        duration = float(input('Duration of simulation: '))
        timestep = 1e-6

        # PART A
        cell = Cell(0, duration, timestep, True, 1)
        for i in tqdm(range(int(duration / timestep) - 1)):
            cell.reactions(i)
        plot_graph(cell.time, cell.x_array, cell.y_array, cell.z_array, 0)
        plt.show()
    else:
        run_b()


def run_b():
    duration = 2.5
    timestep = 1e-6
    animation_timestep = 0.001
    gridsize = 9
    source_position = (4, 4)
    slowfactor = 25
    cell_list = np.empty((gridsize, gridsize), dtype=Cell)

    for i in range(gridsize):
        for j in range(gridsize):
            position = (i, j)
            full = True if position == source_position else False
            new_cell = Cell(position, duration, timestep, full)
            cell_list[i, j] = new_cell

    for y, x in np.ndindex(gridsize, gridsize):
        cell_list[x, y].setup_neighbours(cell_list, source_position, gridsize)

    for i in tqdm(range(int(duration/timestep)-1)):
        for y, x in np.ndindex(gridsize, gridsize):
            cell_list[x, y].reactions(i)
        for y, x in np.ndindex(gridsize, gridsize):
            cell_list[x, y].diffusion(i)

    make_animation(cell_list, timestep, duration, animation_timestep, gridsize, slowfactor)

    plt.show()


if __name__ == '__main__':
    run()
