import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def get_file_data(link) -> list[tuple[float, float, float]]:
    """
    Iterates over each file, extracting file data and saving as a list of tuples
    :param str link: string directing to the folder of files to be extracted
    :return list[tuple[float, float, float]] all_data: all the data in list of tuple form
    """

    file_list = os.listdir(link)
    all_data = []
    for file_name in file_list:

        radius = float(file_name.split('r')[1].split('theta')[0])
        angle = float(file_name.split('theta')[1].split('.out')[0])

        with open(f'{link}/{file_name}', 'r') as f:
            data = f.readlines()
            for line in data:
                if ' SCF Done' == line[:9]:
                    val = float(line.split('E(RHF) =  ')[1].split('     A.U.')[0])
                    break

        all_data.append((radius, angle, val))

    return all_data


def construct_xyz(data) -> tuple:
    """
    Constructs x, y, z arrays from the list of tuples
    :param list[tuple[float, float, float]] data: data from get_file_data
    :return list, tuple: list of x, y, z arrays and tuple of equilibrium coordinates
    """

    unique_radii = sorted(np.unique([a[0] for a in data]))
    unique_angles = sorted(np.unique([b[1] for b in data]))

    mesh = np.ndarray((len(unique_radii), len(unique_angles)))

    for entry in data:
        i = unique_radii.index(entry[0])
        j = unique_angles.index(entry[1])
        mesh[i, j] = entry[2]

    xmin_index, ymin_index = np.unravel_index(np.argmin(mesh), np.shape(mesh))
    equilibrium = (unique_radii[xmin_index], unique_angles[ymin_index], mesh[xmin_index, ymin_index])

    return (unique_radii, unique_angles, mesh), equilibrium


def create_and_scale_axes(xyz_data) -> tuple[plt.figure, plt.axes]:
    """
    Creates a figure and axes and scales it to the size of the plot
    :param tuple[Any, Any, Any] xyz_data: data from construct_xyz
    :return tuple[plt.figure, plt.axes] (fig, axs): The figure and axes
    """

    x, y, z = xyz_data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.set_xlim3d(np.min(x), np.max(x))
    ax.set_ylim3d(np.min(y), np.max(y))
    ax.set_zlim3d(np.min(z), np.max(z))

    return fig, ax


def plot_3d(xyz_data, equilibrium, element) -> None:
    """
    Plots the graphs
    :param list xyz_data: data from construct_xyz
    :param tuple equilibrium: coordinates of equilibrium position
    :param str element: which element's dihydride is being plotted (H2A)
    :return: None
    """

    X, Y, Z = xyz_data
    X, Y = np.meshgrid(X, Y)
    fig, ax = create_and_scale_axes((X, Y, Z))

    ax.plot_surface(X, Y, Z.T, zorder=0, cmap=cm.plasma, edgecolor='black', linewidth=0.1)
    ax.scatter(*equilibrium, color='red', alpha=0.5, zorder=1, lw=0, label='Equilibrium Geometry')
    ax.set_xlabel('Stretch Distance (Å)')
    ax.set_ylabel('Bond Angle (º)')
    ax.set_zlabel('Energy (Hartree)')
    ax.set_title(rf'H$_2${element} Energy Profile')
    ax.legend(loc=(0.75, 1))


def calculate_freq(xyz, equilibrium, element) -> None:
    """
    Prints the stretching frequencies
    :param list xyz: radius, angle and energy arrays
    :param tuple equilibrium: equilibrium coordinates
    :param str element: which element's dihydride is being plotted (H2A)
    :return: None
    """
    r, t, e = xyz

    const_r = e[r.index(equilibrium[0]), :]
    const_t = e[:, t.index(equilibrium[1])]

    ri = r.index(equilibrium[0])
    ti = t.index(equilibrium[1])

    r = [a * 1e-10 for a in r]  # convert radii to m
    t = [a * (np.pi / 180) for a in t]  # convert angles to radians
    const_t = [a * 4.35974e-18 for a in const_t]  # convert energy to joules
    const_r = [a * 4.35974e-18 for a in const_r]

    kt = 2*np.polyfit(t[ti-1:ti+3], const_r[ti-1:ti+3], 2)[0]
    kr = 2*np.polyfit(r[ri-1:ri+3], const_t[ri-1:ri+3], 2)[0]

    m_u = 1.66e-27
    mu_1 = 2 * m_u
    mu_2 = 0.5 * m_u

    eq_r = equilibrium[0] * 1e-10

    v1 = np.sqrt(kr / mu_1) / (2 * np.pi)
    v2 = np.sqrt(kt / (mu_2 * (eq_r * eq_r))) / (2 * np.pi)

    v1 = v1 * 3.33565e-11
    v2 = v2 * 3.33565e-11
    print(f'''==================================================
For H2{element}, the following data was calculated:
Equilibrium bond length: {equilibrium[0]} Å
Equilibrium bond angle: {equilibrium[1]}°
Equilibrium bond energy: {-equilibrium[2].round(1)} Hartree
Symmetric stretching frequency: {v1.round()} cm-1
Bending stretching frequency: {v2.round()} cm-1
==================================================''')


def run_ex2() -> None:
    """
    Run Exercise 2
    :return: None
    """

    for url in ['./H2Ooutfiles', './H2Soutfiles']:

        data = get_file_data(url)

        xyz, equilibrium = construct_xyz(data)

        calculate_freq(xyz, equilibrium, url[4])

        plot_3d(xyz, equilibrium, url[4])

    plt.show()


if __name__ == '__main__':
    run_ex2()
