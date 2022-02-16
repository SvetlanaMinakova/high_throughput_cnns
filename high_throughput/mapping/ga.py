from high_throughput.mapping.chromosome import Chromosome
from high_throughput.eval_table.throughput.communication_penalty import get_communication_penalty
import operator
import random
from models.TaskGraph import TaskGraph
from models.edge_platform import Architecture


class GA:
    """
    Genetic algorithm, that performs efficient mapping of an Application graph (CNN) onto the target platform
        :param app_graph: application task graph, defined as an object of AppGraph class
        :param architecture: platform architecture, defined as an object of Architecture class
        :param time_eval_matrix: matrix [M]x[N] where m in [0, len (processor_types_distinct)-1] represents type of processor,
            available in the platform architecture, n in [0, layers_num] represents layer (node of app_graph),
            matrix[m][n] contains execution time of layer n on processor m
        :param epochs: number of epochs to run GA
        standard GA parameters
        :param population_start_size: number of samples in the start population
        :param selection_percent: percent of population to be selected at every iteration
        :param mutation_probability: chance that a mutation will happen 0=<x<=1
        :param mutation_percent: percent of the samples in the current offspring to mutate
        :param max_no_improvement_epochs: max epochs to continue GA after exec. time have stopped improving
        :param eval_communication (flag): if True, communication between processors will be taken into account
        :param verbose: print details

        GA population init modifications: enforce GA to give preference to one processor
           (e.g. platform GPU) during generation of first offspring of mappings
        :param preferred_proc_id preferred processor id
        :param preset_preferred_proc_probability percent of layers to map on the preferred processor
        :return best found mapping
    """
    def __init__(self, app_graph: TaskGraph, architecture: Architecture, time_eval_matrix, epochs=10,
                 population_start_size=100, selection_percent=50, mutation_probability=0,
                 mutation_percent=10, max_no_improvement_epochs=10,
                 eval_communication=False, verbose=True,
                 preferred_proc_id=-1, preset_preferred_proc_probability=-1):
        
        self.app_graph = app_graph
        self.architecture = architecture

        self.eval_communication = eval_communication
        self.time_eval_matrix = time_eval_matrix
        self.population_start_size = population_start_size
        self.epochs = epochs
        self.selection_percent = selection_percent
        self.mutation_probability = mutation_probability
        self.mutation_percent = mutation_percent
        self.max_no_improvement_epochs = max_no_improvement_epochs

        # GA modifications: enforce GA to give preference to one processor
        # (e.g. platform GPU) during mapping
        self.preferred_proc_id = preferred_proc_id
        self.preset_preferred_proc_probability = preset_preferred_proc_probability
        
        # meta-data
        self.population = []
        self.selected_offspring = []
        self.time_evals = {}
        self.time_evals_sorted = {}
        self.verbose = verbose

    def generate_random_population(self):
        self.population = []
        for i in range(0, self.population_start_size):
            random_chromosome = self.generate_random_chromosome()
            self.population.append(random_chromosome)

    def generate_random_chromosome(self):
        random_chromosome = Chromosome(self.architecture.processors_num, self.app_graph.tasks_num)
        random_chromosome.init_random(self.preferred_proc_id, self.preset_preferred_proc_probability)
        return random_chromosome

    def run(self):
        if self.verbose:
            print("START GA, epochs = ", self.epochs, ", init_offspring: ", self.population_start_size,
                  ", selection:", self.selection_percent, "%", ", mutation probability:", self.mutation_probability)
        cur_epoch = 0
        cur_time = 0
        # we are going to iteratively select top selection_percent chromosomes of current population ...
        chromosomes_to_select = int(self.population.__len__() * self.selection_percent / 100)

        # if 1. best time for population improves for >= no_improvement_epochs ...
        cur = self.population[0]
        cur_time = cur.eval_time(self.architecture.processors_types,
                                 self.architecture.processors_types_distinct,
                                 self.time_eval_matrix)
        best = cur
        best_time = cur_time
        no_improvement_epochs = 0

        # ... and 2. there is something to select, and 3. done epochs < max_epochs,
        while chromosomes_to_select > 0 and cur_epoch < self.epochs:
            self.make_iteration(chromosomes_to_select)
            chromosomes_to_select = int(len(self.population) * self.selection_percent / 100)
            cur_epoch = cur_epoch + 1
            cur = self.population[0]
            cur_time = cur.eval_time(self.architecture.processors_types,
                                     self.architecture.processors_types_distinct,
                                     self.time_eval_matrix)
            best_time = best.eval_time(self.architecture.processors_types,
                                       self.architecture.processors_types_distinct,
                                       self.time_eval_matrix)
            improved = False

            if cur_time < best_time:
                if self.verbose:
                    print("epoch", cur_epoch, " cur time ", cur_time, " < ", "best_time", best_time, "best mapping reset to")
                best = cur
                best_time = cur_time
                if self.verbose:
                    best.print_short()
                improved = True

            if self.verbose:
                print("EPOCH: ", cur_epoch, "epoch best time: ", cur_time, "GA best time: ", best_time)
                print("population ", self.population.__len__(), "time direct_measurements: ", self.time_evals.__len__())

            if improved:
                no_improvement_epochs = 0
            else:
                no_improvement_epochs = no_improvement_epochs + 1

            if no_improvement_epochs == self.max_no_improvement_epochs:
                if self.verbose:
                    print("ALGORITHM FINISHED ON EPOCH", cur_epoch, " ,NO IMPROVEMENT FOR", self.max_no_improvement_epochs, "EPOCHS")
                # partitions = get_all_partitions(best.mapping, self.app_graph.tasks_adjacent_list)
                if self.verbose:
                    best.print_short()
                return best.mapping
                # return partitions

            if chromosomes_to_select == 0:
                if self.verbose:
                    print("ALGORITHM FINISHED ON EPOCH", cur_epoch, " , POPULATION SIZE: ", self.population.__len__())

            if cur_epoch == self.epochs:
                if self.verbose:
                    print("ALGORITHM FINISHED, MAX EPOCHS: ", cur_epoch, " ACHIEVED: ")

        # partitions = get_all_partitions(best.mapping, self.app_graph.tasks_adjacent_list)
        if self.verbose:
            print("ALGORITHM FINISHED WITH ACHIEVED EXEC TIME", best_time)
        if self.verbose:
            best.print_short()
            fin_eval = best.eval_time(self.architecture.processors_types, self.architecture.processors_types_distinct, self.time_eval_matrix)
            if self.verbose:
                print("fin eval_table time: ", fin_eval)
        return best.mapping
        # return partitions

    def make_iteration(self, chromosomes_to_select):
        """" Make GA iteration:
         - select chromosomes_to_select from current population
         - crossover and get a child for every couple [x, x+1] in selected chromosomes
         - set current population = selected chromosomes + their children
         - mutate mutation_percent of population with probablity = mutation_probablity
        """

        # select top chromosomes_to_select from current population
        self.select(chromosomes_to_select)

        # crossover every couple [x, x+1] in selected offspring, and add children into population
        for i in range(0, int(self.selected_offspring.__len__()/2)):
            parent1 = self.selected_offspring[(2 * i)]
            parent2 = self.selected_offspring[(2 * i + 1)]
            child = self.crossover(parent1, parent2)
            self.selected_offspring.append(child)

        # set current offspring as selected offspting
        self.population = self.selected_offspring

        # make mutation_percent of our population more diverse with probability mutation_probability
        self.mutate()

    """
    GA operators
    """

    def select(self, chromosomes_to_select: int):
        """
        Selection: select top chromosomes_to_select from current population
        :param chromosomes_to_select top chromosomes to select
        """
        self.selected_offspring = []
        self.time_evals = {}
        self.time_evals_sorted = {}

        chromosome_id = 0
        # evaluate current population
        for chromosome in self.population:
            time = chromosome.eval_time(self.architecture.processors_types, self.architecture.processors_types_distinct, self.time_eval_matrix)

            if self.eval_communication:
                communication_time = get_communication_penalty(self.app_graph, self.architecture, chromosome.mapping)
                time = time + communication_time

            self.time_evals[chromosome_id] = time
            chromosome_id = chromosome_id + 1

        # sort evaluations by value
        self.time_evals_sorted = sorted(self.time_evals.items(), key=operator.itemgetter(1))

        # select top selection_percent chromosomes of current population
        for i in range(chromosomes_to_select):
            chromosome_id = self.time_evals_sorted[i][0]
            self.selected_offspring.append(self.population[chromosome_id])

    """
    Mutation: make mutation_percent of our population more diverse with probability mutation_probability
    """

    def mutate(self):
        if self.mutation_probability == 0 or self.mutation_percent == 0:
            return

        # roll a dice: is there a mutation going to happen?
        random_chance = random.uniform(0, 1)  # Random float x, 0 <= x < 1
        # print("Mutation: random chance: ", random_chance)

        # mutate
        if random_chance <= self.mutation_probability:
            # at least one chromosome to mutate
            chromosomes_to_mutate = max((int)(self.mutation_percent/100 * len(self.population)), 1)
            # print("Mutate ", chromosomes_to_mutate, "in current offspring of len", self.population.__len__())
            for i in range(chromosomes_to_mutate):
                random_chromosome_id = random.randint(0, self.population.__len__()-1)
                random_chromosome = self.population[random_chromosome_id]
                random_chromosome.mutate()

        # no mutatuion

    def crossover(self, mapping1, mapping2):
        """
        Crossover: exchange halves of parent chromosomes mapping 1 and mapping 2
        """
        child_mapping = Chromosome(mapping1.processors_num, mapping1.tasks_num)
        # genes of first parent: range1 = [0, (layers_num / 2)]
        for l in range(0, int(mapping1.tasks_num / 2)):
            proc_id = mapping1.get_proc(l)
            # print("mapping 1, layer ", l, "proc = ", proc_id)
            child_mapping.mapping[proc_id].append(l)
        # genes of second parent: range1 = [0, (layers_num / 2 - 1)]
        for l in range(int(mapping1.tasks_num / 2), mapping1.tasks_num):
            proc_id = mapping2.get_proc(l)
            # print("mapping 2, layer ", l, "proc = ", proc_id)
            child_mapping.mapping[proc_id].append(l)

        return child_mapping

    """ 
    Print functions
    """
    def print_population(self):
        crhomosome_id = 0
        for chromosome in self.population:
            print("CRHOMOSOME: ", crhomosome_id)
            chromosome.print(self.architecture.processors, self.app_graph.tasks)
            crhomosome_id = crhomosome_id + 1

    def print_population_short(self):
        chromosome_id = 0
        for chromosome in self.population:
            print("CRHOMOSOME: ", chromosome_id)
            chromosome.print_short()
            chromosome_id = chromosome_id + 1

    def print_population_short_timed(self):
        crhomosome_id = 0
        for chromosome in self.population:
            print("CRHOMOSOME: ", crhomosome_id, "time: ",
                  chromosome.eval_time(self.architecture.processors_types,
                                       self.architecture.processors_types_distinct,
                                       self.time_eval_matrix))
            chromosome.print_short()
            crhomosome_id = crhomosome_id + 1

