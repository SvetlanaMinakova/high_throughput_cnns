
"""
Creates dnn partitioning from dnn mapping (TEST VERSION)
"""


def get_all_partitions_new(mapping, tasks_adjacent_list, verbose=False):
    """
    Obtain all partitions on all processors.
    New version: should resolve cyclic dependencies
    """
    all_partitions = []
    e_out = [ ] # outer connections
    proc_id = 0
    for proc_tasks in mapping:
        partitions = get_partitions(proc_tasks, tasks_adjacent_list)
        all_partitions.append(partitions)

        if verbose:
            print("proc", proc_id, partitions)

        proc_id = proc_id + 1
    return all_partitions


def get_partitions(proc_tasks, tasks_adjacent_list):
    #print(proc_tasks)

    #sort layers in traverse order
    proc_tasks.sort()

    partitions = []
    temp_queue = []
    visited = []

    for l in proc_tasks:
        if not visited.__contains__(l):
            #get layer mapped on proc

            #start new partition
            cur_partition = []

            ####################################################
            # process the header (input_examples) layer of the partition

            cur_partition.append(l)
            visited.append(l)

            outs = layer_outputs(l, tasks_adjacent_list)

            #print(outs)
            for out in outs:
                if is_mapped_on_proc(out, proc_tasks):
                    if out not in visited:
                        temp_queue.append(out)

            ####################################################
            # process non-header (hidden) layers of the partition

            #while there are elements in the temp queue
            while temp_queue:
                # get non-header layer from the queue
                l = temp_queue.pop(0)

                # only the header layer is allowed to have exteral inputs.
                # if a non-header layer has external inputs, finish parition and
                # declare non-header layer as a header layer of the next partition
                if has_external_inputs(l, tasks_adjacent_list, proc_tasks):
                    break

                cur_partition.append(l)
                visited.append(l)

                # find layer outputs
                outs = layer_outputs(l, tasks_adjacent_list)
                # print(outs)

                # if outputs are mapped on the same proc
                for out in outs:
                    if is_mapped_on_proc(out, proc_tasks):
                        if out not in visited:
                            temp_queue.append(out)

            partitions.append(cur_partition)
            #print(partitions)


    return partitions


def layer_outputs(layer_id, adjacent_list):
    """
    Get layer outputs
    :param layer_id layer id
    :param adjacent_list: neural net graph connections, represented as adjacent connections list
    """
    return adjacent_list[layer_id]


def has_external_inputs(layer, adjacent_list, proc_tasks):
    """
    Checks, if layer has external inputs (inputs, mapped on other processors)
    :param layer layer id
    :param adjacent_list: neural net graph, represented as adjacent connections list
    :param proc_tasks : list of ids of tasks, mapped on the processor, where layer l is mapped
    :return True, if layer has external input_examples connections and False otherwise
    """
    inputs = get_inputs(layer, adjacent_list)
    for inp in inputs:
        if inp not in proc_tasks:
            return True
    return False


def get_inputs(layer_id, adjacent_list):
    """
    Get layer outputs
    :param layer_id layer id
    :param adjacent_list: neural net graph connections, represented as adjacent connections list
    """
    inputs_list = []

    for node_id in range(0, len(adjacent_list)):
        outputs_list = adjacent_list[node_id]
        if layer_id in outputs_list:
            inputs_list.append(node_id)

    return inputs_list


def is_mapped_on_proc(layer_id, proc_tasks):
    """
    Checks if a node(layer) is mapped on a processor
    :param layer_id : node(layer) id
    :param proc_tasks : list of ids of tasks, mapped on the processor
    :return True, if a node(layer) is mapped on a processor and False otherwise
    """
    return layer_id in proc_tasks

