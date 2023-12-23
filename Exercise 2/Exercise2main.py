import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


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


def create_and_scale_axes(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.set_xlim3d(min(data[0]), max(data[0]))
    ax.set_ylim3d(min(data[1]), max(data[1]))
    ax.set_zlim3d(min(data[2]), max(data[2]))

    return fig, ax


def plot_3d(ax, sorted_array, equilibrium, element):
    radii, angles, energies = sorted_array

    # creates a matrix grid of the correct size for ax.plot_surface
    x = np.unique(radii)
    y = np.unique(angles)
    X, Y = np.meshgrid(x, y)

    # reshape the energy values into the same shape as the XY grid
    Z = np.asarray(energies).reshape((len(x), len(y))).transpose()

    # plot the surface and make them pretty
    ax.plot_surface(X, Y, Z, zorder=0, cmap=cm.plasma)
    ax.scatter(*equilibrium, color='red', alpha=0.5, zorder=1, lw=0, label='Equilibrium Geometry')
    ax.set_xlabel('Stretch Distance (Å)')
    ax.set_ylabel('Bond Angle (º)')
    ax.set_zlabel('Energy (Hartree)')
    ax.set_title(rf'H$_2${element} Energy Profile')
    ax.legend(loc=(0.75, 1))


def calculate_freq(sorted_values, equilibrium):
    radii, angles, energies = sorted_values
    relative_radii = [(radius-equilibrium[0])*1e-10 for radius in radii]  # in m
    relative_angles = [(angle-equilibrium[1])*(np.pi/180) for angle in angles]  # in radians
    relative_energies = [energy*4.3597e-18 for energy in energies]  # in joules
    tuple_list = list(zip(relative_radii, relative_angles, relative_energies))

    # construct list of energies with constant radius (and close to equilibrium angle)
    eq_radii = [coords for coords in tuple_list if coords[0] == 0 and abs(coords[1]) < 0.4]
    const_radii, var_angles, var_energies = list(zip(*eq_radii))
    print(np.polyfit(var_angles, var_energies, 2))
    k_theta = np.polyfit(var_angles, var_energies, 2)[0]*2

    # construct list of energies with constant angle (and close to equilibrium radius)
    eq_angle = [coords for coords in tuple_list if coords[1] == 0 and abs(coords[0]) < 0.2e-10]
    var_radii, const_angle, var_energies = list(zip(*eq_angle))
    print(np.polyfit(var_radii, var_energies, 2))
    k_r = np.polyfit(var_radii, var_energies, 2)[0]*2

    m_u = 1.66e-27
    mu_1 = 2 * m_u
    mu_2 = 0.5 * m_u
    eq_r = equilibrium[0] * 1e-10  # TODO workout scalefactors?

    print(k_theta, ",", k_r, ",", k_r/k_theta)

    v1 = np.sqrt(k_r / mu_1) / (2 * np.pi)
    v2 = np.sqrt(k_theta / (mu_2 * (eq_r ** 2))) / (2 * np.pi)

    #  v1 = v1 * 5.03e22
    #  v2 = v2 * 5.03e22
    return v1, v2


def run():
    for url in ['./H2Ooutfiles', './H2Soutfiles']:

        # cache data to prevent re-parsing
        cache_file = f'{url[2:5]}temp'
        if os.path.exists(f'{cache_file}.npy'):
            data = np.load(f'{cache_file}.npy')
        else:
            data = get_file_data(url)
            np.save(cache_file, data)

        # sort data to fix issues with os imports
        sorted_array, equilibrium = sort_data(data)

        v1, v2 = calculate_freq(sorted_array, equilibrium)
        print(v1, v2)

        # create axes and plot
        fig, ax = create_and_scale_axes(sorted_array)
        plot_3d(ax, sorted_array, equilibrium, url[4])

    plt.show()


if __name__ == '__main__':
    run()
