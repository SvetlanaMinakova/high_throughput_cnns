from high_throughput.mapping.chromosome import Chromosome
import random
import json

"""
Creates dnn dnn_partitioning from dnn mapping
"""


def get_all_partitions(mapping, tasks_adjacent_list, verbose = False):
    """
    Obtain all partitions on all processors
    """
    all_partitions = []
    proc_id = 0
    for proc_tasks in mapping:
        partitions = get_partitions(proc_tasks, tasks_adjacent_list)
        all_partitions.append(partitions)

        if verbose:
            print("proc", proc_id, partitions)

        proc_id = proc_id + 1
    return all_partitions


def get_partitions(proc_tasks, tasks_adjacent_list):
    """
    Get partitions on processor.
    :param proc_tasks: array [int a, int b, int c] proc_tasks,
       representing mapping of layers a,b,c on processor proc

    :param tasks_adjacent_list adjacent_list of layers output
       connections with len = number of nodes on graph, len[i] = number of
       output connections of graph node i
       E.g. list [[1, 2], [3], [4], [5], [5], [6], []] represents DNN graph

        1 - 3
      /       \
    0          5 - 6
      \       /
        2 - 4
    """
    # print(proc_tasks)

    # sort layers in traverse order
    proc_tasks.sort()

    partitions = []
    temp_queue = []
    visited = []

    for layer in proc_tasks:
        if layer not in visited:
            # get layer mapped on proc
            # start new partition
            cur_partition = []

            ####################################################
            # process the header (input_examples) layer of the partition

            cur_partition.append(layer)
            visited.append(layer)

            outs = layer_outputs(layer, tasks_adjacent_list)

            # print(outs)
            for out in outs:
                if mapped_on_the_proc(out, proc_tasks):
                    if out not in visited:
                        temp_queue.append(out)

            ####################################################
            # process non-header (hidden) layers of the partition

            # while there are elements in the temp queue
            while len(temp_queue) > 0:
                # get non-header not-visited layer from the queue
                layer = temp_queue.pop(0)
                if layer in visited:
                    break

                # only the header layer is allowed to have exteral inputs.
                # if a non-header layer has external inputs, finish parition and
                # declare non-header layer as a header layer of the next partition
                # if has_external_inputs(l, tasks_adjacent_list, proc_tasks):
                    # print(l, " has external inputs")
                #    break

                # find layer outputs
                outs = layer_outputs(layer, tasks_adjacent_list)
                # print(outs)


                # one partition can only have one output breaks merge optimization in tensorrt!
                # if outs.__len__() > 1:
                #    break

                cur_partition.append(layer)
                visited.append(layer)

                # if outputs are mapped on the same proc
                for out in outs:
                    if mapped_on_the_proc(out, proc_tasks):
                        if out not in visited:
                            temp_queue.append(out)

                if has_external_outputs(layer, tasks_adjacent_list, proc_tasks):
                    print(layer, " has external outputs")
                    # break

            partitions.append(cur_partition)
    return partitions


def layer_outputs(layer_id, adjacent_list):
    """
    Get layer outputs
    :param layer_id layer id
    :param adjacent_list: neural net graph, represented as adjacent connections list 
    """
    try:
        return adjacent_list[layer_id]
    except Exception:
        print("adjacent list for layer", layer_id, "not found")


def has_external_inputs(layer_id, adjacent_list, proc_tasks):
    """
    Checks, if layer has external input_examples connections
    :param layer_id layer id
    :param adjacent_list: neural net graph, represented as adjacent connections list
    :param proc_tasks : list of ids of tasks, mapped on the processor, where layer l is mapped
    :return True, if layer has external input_examples connections and False otherwise
    """
    inputs = layer_inputs(layer_id, adjacent_list)
    if len(inputs) > 1:
        return True
    for inp in inputs:
        if inp not in proc_tasks:
            return True
    return False


def has_external_outputs(task_id, adjacent_list, proc_tasks):
    """
    Checks, if layer has external output connections
    :param task_id task (layer) id
    :param adjacent_list: neural net graph, represented as adjacent connections list
    :param proc_tasks : list of ids of tasks, mapped on the processor, where layer is mapped
    :return True, if layer has external input_examples connections and False otherwise
    """
    outputs = layer_outputs(task_id, adjacent_list)
    for outp in outputs:
        if outp not in proc_tasks:
            return True
    return False


def task_external_outputs(task_id, adjacent_list, proc_tasks):
    """
    Checks, if layer has external output connections
    :param task_id layer id
    :param adjacent_list: neural net graph, represented as adjacent connections list
    :param proc_tasks : list of ids of tasks, mapped on the processor, where layer is mapped
    :return True, if layer has external input_examples connections and False otherwise
    """
    outputs = layer_outputs(task_id, adjacent_list)
    external_outs = []
    for outp in outputs:
        if outp not in proc_tasks:
            external_outs.append(outp)
    return external_outs


def task_external_inputs(layer_id, adjacent_list, proc_tasks):
    """
    Get layer has external inputs (connected layers, mapped on other processors)
    :param layer_id layer id
    :param adjacent_list: connections between cnn layers, represented as adjacent connections list
    :param proc_tasks : list of ids of tasks, mapped on the processor, where layer layer_id is mapped
    :return True, if layer has external input_examples connections and False otherwise
    """
    inputs = layer_inputs(layer_id, adjacent_list)
    external_inputs = []
    for inp in inputs:
        if inp not in proc_tasks:
            external_inputs.append(inp)
    return external_inputs


def layer_inputs(layer_id, adjacent_list):
    """
    Get layer inputs
    :param layer_id layer id
    :param adjacent_list: list of adjacent connections in the application graph
    """
    inputs_list = [ ]

    for node_id in range (0, adjacent_list.__len__()):
        outputs_list = adjacent_list[node_id]
        if layer_id in outputs_list:
            inputs_list.append(node_id)

    return inputs_list


def mapped_on_the_proc(layer_id, proc_tasks):
    """
    Checks if a node(layer) is mapped on a processor
    :param layer_id : node(layer) id
    :param proc_tasks : list of ids of tasks, mapped on the processor
    :return True, if a node(layer) is mapped on a processor and False otherwise
    """
    return layer_id in proc_tasks


# randomly swap 2x positions of ids in layers
def random_order_swap(layers):
    layers_num = layers.__len__()
    pos1 = random.randint(0, layers_num - 1)  # get random position 1
    pos2 = random.randint(0, layers_num - 1)  # get random position 2
    tmp = layers[pos1]
    layers[pos1] = layers[pos2]
    layers[pos2] = tmp


"""
Print mapping to console
"""
def print_mapping(partitions, processor_labels, task_labels):
    partition_id = 0
    for proc_partitions in partitions:
        print(processor_labels[partition_id], "{")
        for partition in proc_partitions:
            labeled_partition = []
            for task in partition:
                labeled_partition.append(task_labels[task])
            print(labeled_partition)

        print("}")
        partition_id = partition_id + 1


def print_expected_exec_time(partitions, eval_table, architecture):
    """
    Print expected execution time to console
    """
    for proc_id in range(len(partitions)):
        proc_type = architecture.processors_types[proc_id]
        proc_type_id_distinct = get_distinct_processor_type_id(proc_type, architecture.processors_types_distinct)
        proc_time = 0
        proc_partitions = partitions[proc_id]
        for partition in proc_partitions:
            for task in partition:
                task_time = eval_table[proc_type_id_distinct][task]
                proc_time += task_time

        print(architecture.processors[proc_id], " expected time = ", proc_time)


def get_distinct_processor_type_id(processor_type, processor_types_distinct):
    """
    Get id of given processor type in distinct processor type list
    """
    for ptd_id in range(processor_types_distinct.__len__()):
        ptd = processor_types_distinct[ptd_id]
        if ptd == processor_type:
            return ptd_id
    raise Exception ("processor_type", processor_type, " is not found in list ", processor_types_distinct)


class SetEncoder(json.JSONEncoder):
    """
    JSON encoder for custom objects
    """
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)

"""
Save list in a .json_converters file
@ l list to be saved in a .json_converters file
@ filepath path to the .json_converters file
"""
def save_list_as_json(l, filepath):
    with open(filepath, 'w') as outfile:
        json.dump(l, outfile, cls=SetEncoder, indent=4)


"""
Save partitions as a .json_converters file
@ partitions : partitions, obtained from a genetic algorithm
@ processor labels: names of the processors
@ task_labels: names of CNN layers (tasks)
"""
def save_partitions_to_file(partitions, processor_labels, task_labels, filepath):
    json_obj_list = [ ]

    for taskId in range(task_labels.__len__()):
        proc_id = find_partition_start_proc(partitions, taskId)
        if proc_id == -1:
            pass
        else:
            #print(task_labels[taskId], "is a partition start on proc", proc_id)
            proc = partitions[proc_id]
            partition = find_partition(proc, taskId)
            json_p_obj = { }
            json_p_obj ["proc_id"] = proc_id
            json_p_obj["proc_name"] = processor_labels[proc_id]
            p_task_list = [ ]

            for task in partition:
                p_task_list.append(task_labels[task])

            json_p_obj["layers"] = p_task_list
            json_obj_list.append(json_p_obj)

    save_list_as_json(json_obj_list, filepath)
    print("dnn_partitioning is saved in file", filepath)

"""
Print partitions to convole
@ partitions : partitions, obtained from a genetic algorithm
@ processor labels: names of the processors
"""
def print_partitions(partitions, processor_labels, task_labels):
    print("[")
    for taskId in range(task_labels.__len__()):
        proc_id = find_partition_start_proc(partitions, taskId)
        if proc_id == -1:
            # print("partitions printout error: proc id = -1")
            pass
        else:
            #print(task_labels[taskId], "is a partition start on proc", proc_id)
            proc = partitions[proc_id]
            partition = find_partition(proc, taskId)

            print(" {")
            print("   \"proc_id\": ", proc_id, ",")
            print("   \"proc_name\": \"" + processor_labels[proc_id] + "\",")
            print("   \"layers\": [")
            ptcid = 0 #partition tasks comma id
            for task in partition:
                str_to_print = "     \"" + task_labels[task] + "\""
                #add comma to all task strs except of the last one
                if(ptcid < partition.__len__() - 1):
                    str_to_print = str_to_print + ","
                ptcid = ptcid + 1
                print(str_to_print)
            print("   ] ")
            print(" },")

    print("]")


def find_partition_start_proc(partitions, taskId):
    # print("find partition start proc: taskId=", taskId)
    proc_id = 0
    for proc_partitions in partitions:
        for partition in proc_partitions:
            #task found as partition start
            if partition[0] == taskId:
                return proc_id
            #task found, but is is not a partition start
            for i in range(1, partition.__len__()):
                if partition[i] == taskId:
                    return -1
        proc_id = proc_id + 1
    #task not found
    return -1

def find_partition(proc, startTaskId):
    for proc_tasks in proc:
        if proc_tasks[0] == startTaskId:
            return proc_tasks
    return None


"""
test dnn_partitioning creator
"""
def test():

    #graph nodes in traverse order
    layers = [0, 1, 2, 3, 4, 5, 6]

    #graph connectivity - outputs list
    tasks_adjacent_list = [[1, 2], [3], [4], [5], [5], [6], []]

    #graph connectivity - inputs list
    tasks_reverse_adjacent_list = [[], [0], [0], [1], [2], [3, 4], [5]]

    #task output data writing communication cost
    #always ends with 0 because output layer never writes
    tasks_out_comm_cost = [0.1, 0.25, 0.35, 0.45, 0.55, 0]

    layers_num = len(layers)
    processors = [0, 1]
    processors_num = len(processors)

    ch1 = Chromosome(processors_num, layers_num)
    ch1.init_random()

    proc_tasks = ch1.mapping[0]
    gpu_partitions = get_partitions(proc_tasks, tasks_adjacent_list)
    print("gpu partitions: ")
    print(gpu_partitions)
