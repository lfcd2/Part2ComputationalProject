import os
import matplotlib.pyplot as plt
import numpy as np


def get_file_data(link):
    filelist = os.listdir(link)
    radii, angles, values = [], [], []

    for file_name in filelist:

        # extracts radii and angle
        radius = float(file_name.split('r')[1].split('theta')[0])
        angle = float(file_name.split('theta')[1].split('.out')[0])
        radii.append(radius)
        angles.append(angle)

        # finds the correct line and extract the value (val)
        with open(f'{link}/{file_name}', 'r') as f:
            data = f.readlines()
            for line in data:
                if ' SCF Done' == line[:9]:
                    val = float(line.split('E(RHF) =  ')[1].split('     A.U.')[0])
                    values.append(val)
                    print(f'H2{link[4]}: Stretch = {radius} Å, Bond Angle = {angle}º, Energy = {round(val, 7)} Hartree')

    return radii, angles, values


def create_and_scale_axes(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.set_xlim3d(min(data[0]), max(data[0]))
    ax.set_ylim3d(min(data[1]), max(data[1]))
    ax.set_zlim3d(min(data[2]), max(data[2]))

    return fig, ax


def sort_data(data):
    # sorts the list of data points by angles to fix os bugs (it puts 100-160 before 70-99)
    list_of_data_points = list(zip(*data))
    list_of_data_points.sort(key=lambda a: a[1])

    # sorting by angles merged the radii entries, so this sorts them back out
    master_list = []
    for unique_radius in np.unique(data[0]):
        rlist = []
        for entry in list_of_data_points:
            if entry[0] == unique_radius:
                rlist.append(entry)
        master_list += rlist

    # read equilibrium geometry
    equilibrium = min(master_list, key=lambda a: a[2])

    # unzips the list from tuples
    sorted_list = list(zip(*master_list))

    return sorted_list, equilibrium


def run():
    for url in ['./H2Ooutfiles', './H2Soutfiles']:
        data = get_file_data(url)
        fig, ax = create_and_scale_axes(data)
        sorted_array, equilibrium = sort_data(data)
        # TODO Fit a curve in each direction around the equilibrium to find the stretching frequencies
        radii, angles, energies = sorted_array

        # creates a matrix grid of the correct size for ax.plot_surface
        x = np.unique(radii)
        y = np.unique(angles)
        X, Y = np.meshgrid(x, y)

        # reshape the energy values into the same shape as the XY grid
        Z = np.asarray(energies).reshape((len(x), len(y))).transpose()

        # plot the surface and make them pretty
        ax.plot_surface(X, Y, Z, zorder=0)
        ax.scatter(*equilibrium, color='red', alpha=0.5, zorder=1, lw=0, label='Equilibrium Geometry')
        ax.set_xlabel('Stretch Distance (Å)')
        ax.set_ylabel('Bond Angle (º)')
        ax.set_zlabel('Energy (Hartree)')
        ax.set_title(rf'H$_2${url[4]} Energy Profile')
        ax.legend(loc=(0.75, 1))

    plt.show()


if __name__ == '__main__':
    run()
