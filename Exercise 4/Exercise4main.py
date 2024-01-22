import numpy as np
import os
import math


DELTA = 1e-11
LAMBDA = 1e-3


class Vec3d:
    def __init__(self, pos=(0, 0, 0)):
        self.r = list(pos)

    def __sub__(self, other):
        ret = Vec3d()
        for i in range(3):
            ret.r[i] = self.r[i] - other.r[i]
        return ret

    def __add__(self, other):
        ret = Vec3d()
        for i in range(3):
            ret.r[i] = self.r[i] + other.r[i]
        return ret

    def length(self):
        return math.sqrt(self.r[0]*self.r[0] + self.r[1]*self.r[1] + self.r[2]*self.r[2])

    def distance(self, other):
        return (self - other).length()

    def location(self):
        return self.r

    def update_index(self, i, new):
        self.r[i] = new


class System:
    def __init__(self, positions, re):
        self.positions = positions.copy()
        self.offset_positions_positive = positions.copy()
        self.offset_positions_negative = positions.copy()
        self.improved_positions = positions.copy()

        self.re = re

    def potential(self, r):
        return r

    def energy(self, positions):
        U = 0
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions[i+1:]):
                r = pos1.distance(pos2)
                U += self.potential(r)
        return U

    def location(self, i):
        return self.positions[i].r

    def iteration_cycle(self):
        sum_change = 0
        largest_change = 0
        for particle_index, position in enumerate(self.positions):
            for coordinate_index, coord in enumerate(position.location()):

                offset_coord_positive = position.location().copy()
                offset_coord_negative = position.location().copy()
                offset_coord_positive[coordinate_index] += DELTA
                offset_coord_negative[coordinate_index] -= DELTA

                self.offset_positions_positive = self.positions.copy()
                self.offset_positions_negative = self.positions.copy()
                self.offset_positions_positive[particle_index] = Vec3d(offset_coord_positive)
                self.offset_positions_negative[particle_index] = Vec3d(offset_coord_negative)

                deltaE = self.energy(self.offset_positions_positive) - self.energy(self.offset_positions_negative)
                deltaU = deltaE/(2*DELTA)
                change = -1 * LAMBDA * deltaU

                new_coordinate = self.positions[particle_index].location()[coordinate_index] + change
                self.improved_positions[particle_index].update_index(coordinate_index, new_coordinate)

                sum_change += change
                if abs(change) > largest_change:
                    largest_change = change

        self.positions = self.improved_positions.copy()
        return sum_change, largest_change


class LJSystem(System):
    def potential(self, r):
        return 4 * ((r ** -12) - (r ** -6))


class MorseSystem(System):
    def potential(self, r):
        return (1 - np.exp(-r + self.re)) ** 2


def start():
    start_positions, num_of_nuclei = load_positions_from_xyz('input.xyz')

    potential_type = '0'
    while potential_type not in ['lj', 'morse']:
        potential_type = input("Input potential type as 'LJ' or 'Morse' (Note: Morse converges slowly): ").lower()

    re = ''
    if potential_type == 'morse':
        while type(re) != float:
            try:
                re = float(input(r"Input value for re/σ: "))
            except ValueError:
                pass

    reaction_system = LJSystem(start_positions, re)
    if potential_type == 'morse':
        reaction_system = MorseSystem(start_positions, re)

    return reaction_system, num_of_nuclei, potential_type, re


def load_positions_from_xyz(filename):
    try:
        with open(filename, 'r') as f:
            raw_data = f.readlines()
            clean_lines = [line.strip().split(' ', 1)[1] for line in raw_data[2:]]
            header = [line.strip() for line in raw_data[:2]]

            num_of_nuclei = int(header[0])

            positions = []
            for line in clean_lines:
                positions.append(Vec3d([float(value) for value in line.split(' ')]))

    except ValueError:
        raise Exception('Invalid XYZ input')

    return positions, num_of_nuclei


def converge(system):

    total_change, indi_change, iter_count = 1, 1, 0
    while abs(total_change) > 1e-9 or abs(indi_change) > 1e-8:
        total_change, indi_change = system.iteration_cycle()
        iter_count += 1
        if iter_count % 500 == 0:
            print(f'Iteration {iter_count} completed')

    return system


def save_output(positions):
    with open('output.xyz', 'w') as f:
        f.write('7\nOutput\n')
        for pos in positions:
            f.write('C ')
            for a in pos.location():
                f.write(str(a))
                f.write(' ')
            f.write('\n')


def finish(reaction_system, num_of_nuclei, potential_type):

    energy_unit = 'ε' if (potential_type == 'lj') else 'De'
    print(f'Final Energy = {reaction_system.energy(reaction_system.positions)} {energy_unit}')

    for i in range(num_of_nuclei):
        position1 = reaction_system.positions[i]
        for j in range(i + 1, num_of_nuclei):
            position2 = reaction_system.positions[j]
            distance = position1.distance(position2)
            print(f'Atom A: {i}, Atom B: {j}, Distance: {round(distance, 3)}σ')

    save_output(reaction_system.positions)


def run():
    clear = 'cls' if os.name == 'nt' else 'clear'

    reaction_system, num_of_nuclei, potential_type, re = start()

    os.system(clear)

    reaction_system = converge(reaction_system)

    os.system(clear)

    finish(reaction_system, num_of_nuclei, potential_type)


if __name__ == '__main__':
    run()
