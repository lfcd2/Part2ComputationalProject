import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class Reaction:

    def __init__(self, species_string, rate):
        """
        Create a reaction profile
        :param str species_string: String defining the reaction, of the form A+2B>B+C+D
        :param float rate: Reaction rate constant
        """

        self.rate = float(rate)

        reagents, products = species_string.split('->')
        reagents = parse_species(reagents)
        self.reagents = {x: reagents.count(x) for x in reagents}
        products = parse_species(products)
        self.products = {x: products.count(x) for x in products}

        self.changes = {x: self.products.get(x, 0) - self.reagents.get(x, 0) for x in
                        set(self.products) | set(self.reagents)}

        self.current_reaction_state = 0


class Cell:

    def __init__(self, duration, timestep, species, conditions):
        """
        Create a cell
        :param float duration: time for which the simulation will run
        :param float timestep: size of iteration delta T
        :param list species: list of unique species in the system
        :param dict conditions: starting conditions of the cell
        """

        self.duration = duration
        self.timestep = timestep
        self.time = np.arange(0, self.duration, self.timestep)

        species_array_dict = {specie: np.zeros(np.shape(self.time)) for specie in species}
        for s in species:
            species_array_dict[s][0] = conditions[s]
        self.state = species_array_dict
        self.species = species

    def iterate(self, i, reactions):
        """
        Iterate over a timestep
        :param int i: time index
        :param list reactions: reactions to be processed
        :return:
        """

        current_values = {s: self.state[s][i] for s in self.species}

        for reaction in reactions:
            reaction_current_state = reaction.rate
            for reagent, power in reaction.reagents.items():
                reaction_current_state *= current_values[reagent] ** power
            reaction.current_reaction_state = reaction_current_state

        for target_species in self.species:
            dTargetSpecies_dt = 0
            for r in reactions:
                dTargetSpecies_dt += r.changes.get(target_species, 0) * r.current_reaction_state
            self.state[target_species][i + 1] = current_values[target_species] + self.timestep * dTargetSpecies_dt


def plot_graph(time, state_dict, to_plot, log):
    """
    Plot the graph of reagents
    :param ndarray time: x-axis
    :param dict state_dict: dict of arrays to be plotted
    :param list to_plot: list of species to plot
    :return:
    """

    fig, ax = plt.subplots()
    for label, data_array in state_dict.items():
        if label in to_plot:
            ax.plot(time, data_array, label=label)
    ax.legend()
    plt.title('Plot of Concentration against time')
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Concentration / M')
    if log:
        ax.set_yscale('log')
    plt.savefig('Part3Output.png', dpi=300)
    plt.show()


def parse_input_file(input_file):
    """
    Parse the input file to get reactions, starting conditions, duration, timestep
    :param str input_file: location of input file
    :return: list reactions, dict starting conditions, float duration,
             float timestep, list unique, list to_plot, bool log
    """

    reactions = []
    start_conditions = {}

    with open(input_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            if line == '' or line[0] == '#':
                pass
            elif '/' in line:
                reaction_species, reaction_rate = line.split('/')
                if '->' not in reaction_species:
                    raise Exception('Invalid Reaction')
                try:
                    reaction_rate = float(reaction_rate)
                except ValueError:
                    raise Exception('Invalid Reaction Rate')
                reactions.append(Reaction(reaction_species, reaction_rate))
            elif 'Duration:' in line:
                duration = line.split(':')[1]
                try:
                    duration = float(duration)
                except ValueError:
                    raise Exception('Invalid Duration')
            elif 'Timestep:' in line:
                timestep = line.split(':')[1]
                try:
                    timestep = float(timestep)
                except ValueError:
                    raise Exception('Invalid Duration')
            elif 'Plot:' in line:
                to_plot = line.split(':')[1].split(',')
            elif 'Logarithmic:' in line:
                log = bool(line.split(':')[1])
            elif ':' in line:
                species, concentration = line.split(':')
                start_conditions[species] = float(concentration)

    unique = sorted(list(set().union(*(reaction.changes.keys() for reaction in reactions))))
    for u in unique:
        if u not in start_conditions.keys():
            raise Exception('Insufficient Starting Conditions or invalid reactions')
    for s in to_plot:
        if s not in unique:
            raise Exception('Plotting variable not in reactions')
    return reactions, start_conditions, duration, timestep, unique, to_plot, log


def parse_species(species):
    """
    Converts string of the form A+2B+C to a list ['A', 'B', 'B', 'C']
    :param str species: reagent string
    :return list species: reagents expressed as a list
    """

    species = species.split('+')
    for a in species:
        if a[0].isnumeric():
            species.pop(species.index(a))
            species = species + list(np.full(int(a[0]), a[1:]))
    return species


def run_ex3gc():
    """Run the program"""

    reactions, start_conditions, duration, timestep, all_unique_species,  to_plot, log = parse_input_file('input.txt')

    cell = Cell(duration, timestep, all_unique_species, start_conditions)

    for i in tqdm(range(int(duration / timestep) - 1)):
        cell.iterate(i, reactions)

    plot_graph(cell.time, cell.state, to_plot, log)


if __name__ == '__main__':
    run_ex3gc()
