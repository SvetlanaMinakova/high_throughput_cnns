import random


class Chromosome:
    """
    Chromosome that represents mapping of a CNN on an edge platform for GA-based
    efficient mapping search
    """
    def __init__(self, processors_num, tasks_num):
        self.mapping = []
        self.processors_num = processors_num
        self.tasks_num = tasks_num

        for i in range(self.processors_num):
            self.mapping.append([])

    def mutate(self):
        random_proc_id = random.randint(0, self.processors_num - 1)  # get random processor
        random_layer_id = random.randint(0, self.tasks_num - 1)  # get random layer
        old_proc_id = self.get_proc(random_layer_id)  # get current mapping of the layer
        self.mapping[old_proc_id].remove(random_layer_id)
        self.mapping[random_proc_id].append(random_layer_id)
        # print("Mutate: move layer ", random_layer_id, "from proc", old_proc_id, "to proc", random_proc_id)

    def clean(self):
        self.mapping = []
        for i in range(self.processors_num):
            self.mapping.append([])

    def init_random(self, preferred_proc_id=-1, preset_preferred_proc_probability=-1):
        """
        Init a chromosome randomly
        GA population init modifications: enforce GA to give preference to one processor
           (e.g. platform GPU) during generation of first offspring of mappings
        :param preferred_proc_id preferred processor id
        :param preset_preferred_proc_probability percent of layers to map on the preferred processor
        :return:
        """
        # no preset processor
        if preset_preferred_proc_probability < 0 or preferred_proc_id < 0:
            for layer_id in range(self.tasks_num):
                random_proc_id = random.randint(0, self.processors_num - 1)  # get random processor
                self.mapping[random_proc_id].append(layer_id)
        else:
            for layer_id in range(self.tasks_num):
                # roll a dice: should we move a layer to gpu?
                random_chance = random.uniform(0, 1)  # Random float x, 0 <= x < 1
                if random_chance <= preset_preferred_proc_probability:
                    # the layer goes to GPU
                    self.mapping[preferred_proc_id].append(layer_id)
                else:
                    # select processor randomly
                    random_proc_id = random.randint(0, self.processors_num - 1)  # get random processor
                    self.mapping[random_proc_id].append(layer_id)

    def print(self, processors_labels, layer_labels):
        for p in range(self.processors_num):
            print(processors_labels[p], " {")
            for layer_id in self.mapping[p]:
                print(layer_labels[layer_id])
            print("}")

    def print_short(self):
        for p in range(self.processors_num):
            print("proc: ", p, "layers: ", self.mapping[p])

    def get_proc(self, layer_id):
        for p in range(self.processors_num):
            if self.mapping[p].__contains__(layer_id):
                return p
        return None

    # Execution time = max (proc1, proc2, ..., procP),  since we assume all the processors to work in parallel
    def eval_time(self, processor_types, distinct_processor_types, times_eval_table):
        times_per_proc = self.eval_times_per_proc(processor_types, distinct_processor_types, times_eval_table)
        max_time = max(times_per_proc)
        return max_time

    # evaluate execution time per processor
    def eval_times_per_proc(self, processor_types, distinct_processor_types, times_eval_table):
        proc_times = []
        for p in range(self.processors_num):
            proc_type = processor_types[p]
            distinct_proc_type_id = get_proc_type_id(proc_type, distinct_processor_types)
            proc_time = 0

            # time per processor = sum (times of tasks executed on this processor),
            # since we assume tasks on one processor to work sequentially
            for layer in self.mapping[p]:
                # find time for node layer, executed on processor p of type proc_type_id
                node_time = times_eval_table[distinct_proc_type_id][layer]
                proc_time = proc_time + node_time  # accumulate time for processor

            proc_times.append(proc_time)
            #print("Processor: ", p, " (type = ", distinct_processor_types[distinct_proc_type_id] , ") evaluated")
        return proc_times


def get_proc_type_id(processor_type, distinct_processor_types):
    """
    Get id of processor type
    :param processor_type: processor type (string)
    :param distinct_processor_types: all distinct processor types
    :return:  id of processor type
    """
    for i in range(len(distinct_processor_types)):
        if processor_type == distinct_processor_types[i]:
            return i
    return None

