from dnn_partitioning.after_mapping.partitioning_creator import layer_outputs, layer_inputs
from models.TaskGraph import get_example_graph


class GroupedGraph:
    def __init__(self, original_graph, grouped_layers):
        self.original_graph = original_graph
        self.merged_graph = None
        #array, where i-th element is an array, containing nodes of unm
        self.layers_mapping = None


def get_task_group(groups, task_id):
    for group in groups:
        if task_id in group:
            return group
    raise Exception("task", task_id, "not found in task groups")



def get_grouped_graph(app_graph, grouped_layers):
    #do not merge first and last group, if they are linear



    grouped_layer_ids = [id for id in range()]

    #group layers
    #for group


def print_groups_with_layer_names(groups, layer_names):
    for group in groups:
        group_layers = []
        for layer_id in group:
            group_layers.append(layer_names[layer_id])
        print(group_layers)


def get_layer_groups(app_graph):
    """
    Groups a dnn multi-input_examples and multi-output layers into blocks with one input_examples and one output
    """
    if not app_graph.jobs_per_task:
        return []

    groups = []
    group = []

    group_start = None
    group_end = None

    layers_to_traverse = [layer_id for layer_id in range(len(app_graph.jobs_per_task))]

    layer = layers_to_traverse.pop(0)
    while layers_to_traverse:
        if is_group_start(layer, app_graph):
            if not group_start:
                group_start = layer
            else:
                group_end = layer
        if is_group_end(layer, app_graph):
            group_end = layer

        #create new group
        if group_start is not None and group_end is not None:
            if layer not in group:
                group.append(layer)
            groups.append(group)
            group = []
            group_start = None
            group_end = None

        group.append(layer)
        layer = layers_to_traverse.pop(0)

    if groups[-1]!=group and len(group)>0:
        groups.append(group)
    #add last layer to the last group
    groups[-1].append(layer)
    separate_last_group(groups, app_graph.tasks_adjacent_list)
    unmerge_first_group(groups)
    unmerge_last_group(groups)

    #clean_duplicates(groups)
    return groups



def unmerge_first_group(groups):
    first_group_reversed = [elem for elem in groups[0]]
    first_group_reversed.reverse()
    groups.remove(groups[0])

    for elem in first_group_reversed:
        groups.insert(0, [elem])


def unmerge_last_group(groups):
    if len(groups) <2:
        return
    last_group_reverse = [elem for elem in groups[-1]]
    last_group_reverse.reverse()
    groups.remove(groups[-1])

    for elem in last_group_reverse:
        groups.append([elem])


def separate_last_group(groups, tasks_adjacent_list):
    if len(groups) <2:
        return

    last_group = groups[-1]
    last_separated_group = [ ]

    #traverse back until we meet multiple inputs processor
    continue_adding = True
    for id in range(len(last_group)):
        layer_id = last_group[len(last_group)- id -1]
        if is_multi_input_layer(layer_id, tasks_adjacent_list):
            continue_adding = False
        else:
            if continue_adding:
                last_separated_group.append(layer_id)

    if len(last_separated_group) < len(last_group):
        for elem in last_separated_group:
            groups[-1].remove(elem)
        groups.append(last_separated_group)



def find_duplicates(groups):
    seen = [ ]
    duplicates = [ ]
    for group in groups:
        for value in group:
            if value in seen:
                duplicates.append(value)
            seen.append(value)
    return duplicates


def clean_duplicates(groups):
    for group_id in range(len(groups)-1):
        this_group = groups[group_id]
        next_group = groups[group_id + 1]

        if this_group[-1] == next_group[0]:
            this_group.remove(this_group[-1])


def is_group_end(layer_id, app_graph):
    """
    Checks if layer ends layers group
    :param layer_id: layer id
    :param app_graph: CNN application graph
    :return: True, if layer starts a new group and false otherwise
    """
    if is_output_layer(layer_id, app_graph.jobs_per_task):
        return True

    if is_multi_input_layer(layer_id, app_graph.tasks_adjacent_list):
        return True

    if is_multi_output_layer(layer_id, app_graph.tasks_adjacent_list):
        return True

    return False


def is_multi_input_layer(layer_id, tasks_adjacent_list):
    inputs = layer_inputs(layer_id, tasks_adjacent_list)
    return len(inputs) > 1


def is_output_layer(layer_id, layers):
    return layer_id == len(layers) - 1



def is_group_start(layer_id, app_graph):
    """
    Checks if layer starts a new group
    :param layer_id: layer id
    :param app_graph: CNN application graph
    :return: True, if layer starts a new group and false otherwise
    """
    if is_input_layer(layer_id):
        return True

    if is_multi_output_layer(layer_id, app_graph.tasks_adjacent_list):
        return True

    return False


def is_input_layer(layer_id):
    return layer_id == 0


def is_multi_output_layer(layer_id, tasks_adjacent_list):
    outputs = layer_outputs(layer_id, tasks_adjacent_list)
    return len(outputs) > 1


def test():
    example_app = get_example_graph()
    groups = get_layer_groups(example_app)
    for group in groups:
        print(group)


#tests()