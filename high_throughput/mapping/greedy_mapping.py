import copy
from high_throughput.eval_table.eval_dnn_latency_from_et import get_sum_proc_times,\
    get_pipeline_execution_proc_time


def map_greedy(app_graph, architecture, time_eval_matrix, start_processor_id=0, verbose=True):
    """
    Greedy mapping algorithm accepts as input_examples a list of CNN inference tasks and spreads the tasks over
    the processors (CPUs and GPUs), available on the platform
    using following steps:

    1. All tasks are mapped on the most powerful processor: GPU (if available) or CPU
    while execution time, associate with processors, is not balanced:
        2.1. Take next task from beginning of the task list. If beneficial
            (leads to overall CNN execution time reduction), map it on least busy processor
        2.2 Take next task from the beginning of the task list. If beneficial
            (leads to overall CNN execution time reduction), map it on least busy processor.
        If no tasks were moved from original processor, leave the loop and finish algorithm

    :param app_graph: application task graph, defined as an object of AppGraph class
    :param architecture: platform architecture, defined as an object of Architecture class
    :param time_eval_matrix: matrix [M]x[N] where m in [0, len (processor_types_distinct)-1]
     represents type of processor, available in the platform architecture,
     n in [0, layers_num] represents layer (node of app_graph), matrix[m][n]
     contains execution time of layer n on processor m
    :param start_processor_id: processor where all the tasks are mapped in the beginning
    :param verbose: print details
    """
    # e.g. layers = ["conv1_relu", "conv2_relu", "FC", "Softmax"]
    layers = app_graph.jobs_per_task
    # e.g. ["CPU0", "CPU1", "CPU2", "CPU3", "CPU4", "GPU"]
    processors = architecture.src_and_dst_processor_types
    start_proc_id = 0
    if start_processor_id != -1:
        start_proc_id = start_processor_id

    mapping = [[] for proc in range(len(processors))]

    # map everything on start_processor (GPU or first cpu)
    task_id = 0
    for task in layers:
        mapping[start_proc_id].append(task_id)
        task_id = task_id + 1

    proc_sum_time = get_sum_proc_times(mapping, architecture, time_eval_matrix)

    continue_migrate = True
    gpu_is_the_bottleneck = True
    if verbose:
        print("CNN execution time on single processor: ", get_pipeline_execution_proc_time(proc_sum_time))
    while continue_migrate:
        most_busy_proc_id = get_most_busy_processor_id(mapping, architecture, time_eval_matrix)
        best_migration = get_most_beneficial_migration(mapping, architecture, app_graph, time_eval_matrix, most_busy_proc_id)

        gpu_is_the_bottleneck = is_gpu_a_bottleneck_after_migration(mapping, architecture, time_eval_matrix, start_processor_id, best_migration)
        if gpu_is_the_bottleneck: # best_migration.beneficial():
            if verbose:
                print("task", best_migration.task_id, "will be migrated from processor", best_migration.src_proc_id,
                      "to processor", best_migration.dst_proc_id)
            migrate_task_continuous(mapping, architecture, app_graph, time_eval_matrix, best_migration, start_processor_id, verbose)
        else:
            continue_migrate = False
            if verbose:
                print("no more beneficial migrations left")

    proc_sum_time = get_sum_proc_times(mapping, architecture, time_eval_matrix)
    if verbose:
        print("CNN execution time after migrations: ", get_pipeline_execution_proc_time(proc_sum_time))

    return mapping


class Migration:
    """
    Class describes time benefit of task migration from source processor to destination processor
    """
    def __init__(self, src_proc_id, dst_proc_id, is_head, task_id=-1, time_benefit=0):
        self.src_proc_id = src_proc_id
        self.dst_proc_id = dst_proc_id
        self.is_head = is_head # if task is taken from the head
        self.task_id = task_id
        self.time_benefit = time_benefit

    def beneficial(self):
        return self.time_benefit >= 0 and self.task_id!=-1


def is_gpu_a_bottleneck_after_migration(mapping,  architecture, time_eval_matrix, gpu_id, migration: Migration):
    """
    Checks if GPU is still a bottleneck after migration
    :param mapping: mapping before migration
    :param architecture: architecture
    :param time_eval_matrix: time evaluation matrix
    :param gpu_id: gpu id
    :param migration: processes migration
    :return: True if GPU is still a bottleneck after migration and False otherwise
    """
    mapping_after_migration = copy.deepcopy(mapping)
    # simulate migration

    task_id = migration.task_id
    # no migration
    if task_id < 0:
        return False

    for proc_mapping in mapping_after_migration:
        if task_id in proc_mapping:
            proc_mapping.remove(task_id)
    mapping_after_migration[migration.dst_proc_id].append(migration.task_id)

    # estimate how exec. times will change
    sum_proc_times = get_sum_proc_times(mapping_after_migration, architecture, time_eval_matrix)
    gpu_is_bottleneck = True
    proc_id = 0
    gpu_time = sum_proc_times[gpu_id]
    for sum_time in sum_proc_times:
        if proc_id != gpu_id and sum_time >= gpu_time:
            gpu_is_bottleneck = False
        proc_id += 1
    return gpu_is_bottleneck


def get_most_busy_processor_id(mapping, architecture, time_eval_matrix):
    """Get id of most busy processor"""
    sum_proc_times = get_sum_proc_times(mapping, architecture, time_eval_matrix)
    most_busy_proc_id = 0
    max_time = sum_proc_times[0]

    for proc_id in range(len(architecture.src_and_dst_processor_types)):
        time = sum_proc_times[proc_id]
        if time > max_time:
            most_busy_proc_id = proc_id

    return most_busy_proc_id


def get_task_migration_benefit(task_id, mapping, architecture, app_graph, time_eval_matrix, src_proc_id, dst_proc_id):

    if migration_leads_to_cycles(task_id, mapping, src_proc_id, dst_proc_id):
        return -1

    # Predict execution time changes, that will occur
    # if the first task on the most busy processor is migrated to the least busy processor"""
    exec_times = get_sum_proc_times(mapping, architecture, time_eval_matrix)
    src_time = exec_times[src_proc_id]
    dst_time = exec_times[dst_proc_id]

    src_proc_type = architecture.get_proc_type_id(src_proc_id)
    dst_proc_type = architecture.get_proc_type_id(dst_proc_id)

    src_comm_penalty = 0 #get_communication_penalty(task_id, src_proc_id, mapping, architecture, app_graph)
    dst_comm_penalty = 0 #get_communication_penalty(task_id, dst_proc_id, mapping, architecture, app_graph)

    next_src_time = src_time - time_eval_matrix[src_proc_type][task_id] - src_comm_penalty
    next_dst_time = dst_time + time_eval_matrix[dst_proc_type][task_id] + dst_comm_penalty


    if next_src_time <= src_time and next_src_time > next_dst_time:
        return src_time - next_src_time
    else: return -1

    #return cur_cnn_exec_time - predicted_cnn_exec_time


def migration_leads_to_cycles(task_id, mapping, src_proc_id, dst_proc_id):
    """
    Check if migration can lead to cycles
    :param task_id:
    :param mapping:
    :param src_proc_id:
    :param dst_proc_id:
    :return:
    """
    dst_proc_tasks = mapping[dst_proc_id]
    # no cycles, if nothing mapped on the destination processor yet
    if len(dst_proc_tasks) == 0:
        return False

    # no cycles, if mapping continues chain
    if (task_id-1) in dst_proc_tasks or (task_id + 1) in dst_proc_tasks:
        return False

    return True


def get_most_beneficial_migration(mapping, architecture, app_graph, time_eval_matrix, most_busy_proc_id):
    """
    Get id of the task to migrate
    :param mapping:
    :param architecture:
    :param time_eval_matrix:
    :param most_busy_proc_id:
    :param least_busy_proc_id:
    :return:
    """
    src_proc_id = most_busy_proc_id
    best_dst_proc_id = 0
    best_migration_task_id = 0

    head_task = mapping[src_proc_id][0]
    tail_task = mapping[src_proc_id][-1]

    time_benefit = -1
    is_head = True
    for dst_proc_id in range(len(architecture.src_and_dst_processor_types)):
        if dst_proc_id != src_proc_id:
            head_migration_time_benefit = get_task_migration_benefit(head_task, mapping, architecture, app_graph,
                                                                     time_eval_matrix, most_busy_proc_id, dst_proc_id)
            if head_migration_time_benefit >= 0 and head_migration_time_benefit > time_benefit:
                best_dst_proc_id = dst_proc_id
                best_migration_task_id = head_task
                time_benefit = head_migration_time_benefit
                is_head = True

            tail_migration_time_benefit = get_task_migration_benefit(tail_task, mapping, architecture, app_graph,
                                                                     time_eval_matrix, most_busy_proc_id, dst_proc_id)
            if tail_migration_time_benefit >= 0 and tail_migration_time_benefit > time_benefit:
                best_dst_proc_id = dst_proc_id
                best_migration_task_id = tail_task
                time_benefit = tail_migration_time_benefit
                is_head = False

    # migrate nothing if time benefit = 0 (there is no time benefit), or time benefit <0 (time would be only worser after migration)
    if time_benefit < 0:
        no_migration = Migration(src_proc_id, src_proc_id, is_head, -1, 0)
        return no_migration

    best_migration = Migration(src_proc_id, best_dst_proc_id, is_head, best_migration_task_id, time_benefit)
    return best_migration


def migrate_task(mapping, migration):
    """
    Migrate task from source to destination processor
    """
    mapping[migration.src_proc_id].remove(migration.task_id)
    mapping[migration.dst_proc_id].append(migration.task_id)


def migrate_task_continuous(mapping, architecture, app_graph, time_eval_matrix, start_migration, gpu_id, verbose=True):
    """
    Migrate task from source to destination processor.
    If the task is a head-task, continue with migration all following tasks as long as migration is beneficial
    If the task is a tail-task, continue with migrating all previous tasks as long as migration is beneficial
    """
    migrate_task(mapping, start_migration)

    if start_migration.is_head:
        migrate_next_tasks(mapping, architecture, app_graph, time_eval_matrix, start_migration, gpu_id, verbose)
    else:
        migrate_prev_tasks(mapping, architecture, app_graph, time_eval_matrix, start_migration, gpu_id, verbose)


def get_next_task(mapping, proc_id, task_id):
    tasks = mapping[proc_id]
    for proc_task_id in tasks:
        if proc_task_id > task_id:
            return proc_task_id
    return None


def get_prev_task(mapping, proc_id, task_id):
    tasks_reverse = [task for task in mapping[proc_id]]
    tasks_reverse.reverse()
    for proc_task_id in tasks_reverse:
        if proc_task_id < task_id:
            return proc_task_id

    return None


def migrate_prev_tasks(mapping, architecture, app_graph, time_eval_matrix, start_migration, gpu_id, verbose=True):
    """
    Migrate all prev tasks between two processors
    :param app_graph: application TaskGraph
    :param mapping: mapping of application tasks onto platform processors
    :param architecture: target architecture
    :param time_eval_matrix: eval matrix
    :param start_migration: if it is a start of migration
    :return:
    """
    prev_task = get_prev_task(mapping, start_migration.src_proc_id, start_migration.task_id)
    if verbose:
        print("prev for task", start_migration.task_id, "is task", prev_task)

    if prev_task is None:
        return

    benefit = get_task_migration_benefit(prev_task, mapping, architecture, app_graph, time_eval_matrix, start_migration.src_proc_id, start_migration.dst_proc_id)
    planned_migration = Migration(start_migration.src_proc_id, start_migration.dst_proc_id, False, prev_task, benefit)

    gpu_is_the_bottleneck_after_migration = is_gpu_a_bottleneck_after_migration(mapping, architecture, time_eval_matrix, gpu_id, planned_migration)
    if gpu_is_the_bottleneck_after_migration: # planned_migration.beneficial():
        migrate_task(mapping, planned_migration)
        if verbose:
            print("prev task", prev_task, "is continuously migrated from proc", planned_migration.src_proc_id, "to proc", planned_migration.dst_proc_id)
        migrate_prev_tasks(mapping, architecture, app_graph, time_eval_matrix, planned_migration, gpu_id, verbose)
    else:
        return


def migrate_next_tasks(mapping, architecture, app_graph, time_eval_matrix, start_migration, gpu_id, verbose=True):
    """
    Migrate all next tasks between two processors
    :param mapping:
    :param architecture:
    :param time_eval_matrix:
    :param start_migration:
    :return:
    """
    next_task = get_next_task(mapping, start_migration.src_proc_id, start_migration.task_id)
    if verbose:
        print("next for task", start_migration.task_id, "is task", next_task)

    if next_task is None:
        return

    benefit = get_task_migration_benefit(next_task, mapping, architecture, app_graph, time_eval_matrix,
                                         start_migration.src_proc_id, start_migration.dst_proc_id)
    planned_migration = Migration(start_migration.src_proc_id, start_migration.dst_proc_id, True, next_task, benefit)

    gpu_is_the_bottleneck_after_migration = is_gpu_a_bottleneck_after_migration(mapping, architecture, time_eval_matrix,
                                                                                gpu_id, planned_migration)
    if gpu_is_the_bottleneck_after_migration:  # planned_migration.beneficial():
        migrate_task(mapping, planned_migration)
        mapping[planned_migration.dst_proc_id].sort()
        if verbose:
            print("next task", next_task, "is continuously migrated from proc",
                  planned_migration.src_proc_id, "to proc", planned_migration.dst_proc_id)
        migrate_next_tasks(mapping, architecture, app_graph, time_eval_matrix, planned_migration, gpu_id, verbose)
    else:
        return


def find_proc(mapping, task_id):
    for proc_id in range(len(mapping)):
        if task_id in mapping[proc_id]:
            return proc_id

