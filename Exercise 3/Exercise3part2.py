import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as anim


class Cell:

    def __init__(self, position, duration, timestep, full=False, number_of_cells=1, save_b=False):
        self.position = position
        self.duration = duration
        self.timestep = timestep
        self.neighbours = []
        X = [0, 0]
        Y = [0, 0]
        Z = [0, 0]
        if full:
            X[0] = (10 ** (-9.8)) * number_of_cells
            Y[0] = (10 ** (-6.52)) * number_of_cells
            Z[0] = (10 ** (-7.32)) * number_of_cells
        self.x_array = X
        self.y_array = Y
        self.z_array = Z
        if not save_b:
            self.time = np.arange(0, self.duration, self.timestep)
            self.x_save = np.zeros(np.shape(self.time))
            self.y_save = np.zeros(np.shape(self.time))
            self.z_save = np.zeros(np.shape(self.time))
        if save_b:
            self.time_save_array = []
            self.save_array = []

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

    def reactions(self):
        k1 = 1.34
        k2 = 1.6e9
        k3 = 8e3
        k4 = 4e7
        k5 = 1
        x, y, z = self.x_array[0], self.y_array[0], self.z_array[0]
        a, b = 0.06, 0.06  # Concentration of A and B is much greater than X,Y,Z so steady state

        self.x_array[1] = x + self.timestep * (k1 * a * y - k2 * x * y + k3 * b * x - 2 * k4 * x * x)
        self.y_array[1] = y + self.timestep * (-k1 * a * y - k2 * x * y + k5 * z)
        self.z_array[1] = z + self.timestep * (k3 * b * x - k5 * z)

    def diffusion(self):
        num_of_neighbours = len(self.neighbours)

        cell_x = self.x_array[1]
        delta_x = cell_x * 1e-15
        cell_y = self.y_array[1]
        delta_y = cell_y * 1e-15
        cell_z = self.z_array[1]
        delta_z = cell_z * 1e-15

        self.x_array[1] = cell_x - delta_x
        self.y_array[1] = cell_y - delta_y
        self.z_array[1] = cell_z - delta_z

        for n in self.neighbours:
            n.x_array[1] += delta_x / num_of_neighbours
            n.y_array[1] += delta_y / num_of_neighbours
            n.z_array[1] += delta_z / num_of_neighbours

    def save_part_a(self, i):
        self.x_save[i] = self.x_array[0]
        self.y_save[i] = self.y_array[0]
        self.z_save[i] = self.z_array[0]

    def save_part_b(self):
        self.save_array.append(self.z_array[0])

    def iter(self):
        self.x_array[0] = self.x_array[1]
        self.y_array[0] = self.y_array[1]
        self.z_array[0] = self.z_array[1]
        self.x_array[1] = 0
        self.y_array[1] = 0
        self.z_array[1] = 0


def plot_graph(time, x_array, y_array, z_array):
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(time, x_array, color='blue', label='X')
    ax.plot(time, y_array, color='green', label='Y')
    ax.plot(time, z_array, color='red', label='Z')
    ax.legend()
    plt.title(f'Plot of Concentration against time for the Oregonator reaction')
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Concentration / M')
    ax.set_xlim(0, time[-1])
    ax.set_ylim(2e-11, 0.002)


def make_animation(cells, animation_timestep, slow):
    fig, (ax, cbar) = plt.subplots(1, 2, width_ratios=([7, 1]))
    norm = mpl.colors.LogNorm(vmin=1e-18, vmax=1e-3)

    def anim_func(frame):
        data = [[c.save_array[frame] for c in cell_row] for cell_row in cells]
        ax.pcolor(data, norm=norm)

    plt.colorbar(mappable=mpl.cm.ScalarMappable(norm=norm), cax=cbar)

    ax.set_title(f'Concentration of Z. Slowed by a factor of {slow}.')
    cbar.set_ylabel(fr'Concentration / mol dm$^-3$')
    fig.tight_layout()
    animation = anim.FuncAnimation(fig, anim_func, interval=animation_timestep*1000*slow,
                                   frames=len(cells[0, 0].save_array))
    animation.save('Part2BOutput.mp4')


def run_ex3p2a() -> None:
    """
    Run Experiment 3 Part 2 A (Oreganator - recreating fig 3.4)
    :return: None
    """

    duration = 90
    timestep = 1e-6

    # PART A
    cell = Cell(0, duration, timestep, True, 1)
    for i in tqdm(range(int(duration / timestep) - 1)):
        cell.reactions()
        cell.save_part_a(i)
        cell.iter()
    plot_graph(cell.time, cell.x_save, cell.y_save, cell.z_save)
    plt.show()


def run_ex3p2b() -> None:
    """
    Run Experiment 3 Part 2 B (Oreganator with Diffusion)
    :return: None
    """

    duration = 60
    timestep = 1e-6
    animation_timestep = 0.001
    gridsize = 9
    source_position = (4, 4)
    slowfactor = 2
    cell_list = np.empty((gridsize, gridsize), dtype=Cell)

    for i in range(gridsize):
        for j in range(gridsize):
            position = (i, j)
            full = True if position == source_position else False
            new_cell = Cell(position, duration, timestep, full, save_b=True)
            cell_list[i, j] = new_cell

    for y, x in np.ndindex(gridsize, gridsize):
        cell_list[x, y].setup_neighbours(cell_list, source_position, gridsize)

    for i in tqdm(range(int(duration/timestep)-1)):
        for y, x in np.ndindex(gridsize, gridsize):
            cell_list[x, y].reactions()
        for y, x in np.ndindex(gridsize, gridsize):
            cell_list[x, y].diffusion()
        for y, x in np.ndindex(gridsize, gridsize):
            cell_list[x, y].iter()
        if round(i % (animation_timestep/timestep), 5) == 1:
            for y, x in np.ndindex(gridsize, gridsize):
                cell_list[x, y].save_part_b()

    make_animation(cell_list, animation_timestep, slowfactor)

    plt.show()


if __name__ == '__main__':

    inp = input('Part A or B:')
    if inp.upper() == 'A':
        run_ex3p2a()
    else:
        run_ex3p2b()
