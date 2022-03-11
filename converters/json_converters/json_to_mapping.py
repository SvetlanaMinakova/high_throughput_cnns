import json


def json_to_mapping(path, task_graph, architecture):
    """
    Reads mapping of a DNN onto target edge platform saved as a .json file
    :param path: path to mapping of a DNN onto target edge platform saved as a .json file
    :param task_graph: task-graph of a DNN
    :param architecture: target hardware platform
    :return:
    """
    with open(path, 'r') as file:
        if file is None:
            raise FileNotFoundError
        else:
            json_mapping = json.load(file)
            # create new empty mapping
            mapping = [[] for proc_id in range(len(architecture.processors))]
            # fill mapping according to .json file
            for proc_name in json_mapping:
                # determine processor
                proc_id = get_proc_id(architecture.processors, proc_name)
                # determine tasks (within the task graph ) executed by the processor
                proc_job_names = json_mapping[proc_name]
                proc_task_ids = []
                for job_name in proc_job_names:
                    task_id = get_task_id(task_graph, job_name)
                    if task_id not in proc_task_ids:
                        proc_task_ids.append(task_id)
                # sort task ids
                proc_task_ids = sorted(proc_task_ids)
                # specify task ids
                mapping[proc_id] = proc_task_ids
            return mapping


def get_proc_id(proc_names, proc_name):
    """
    Determine processor id using processor name
    and the list of processor names, sorted by processor id
    :param proc_names: list of processor names, sorted by processor id
    :param proc_name: processor id
    :return: processor id (int)
    """
    for i in range(len(proc_names)):
        if proc_names[i] == proc_name:
            return i
    raise Exception("Mapping parsing error: processor " + proc_name +
                    " not found in platform processors " + str(proc_names))


def get_task_id(task_graph, job_name):
    """
    Determine id of a task in the task graph, using name of a job within the task graph
    For DNNs job_name = name of a DNN layer
    :param task_graph: task-graph of a DNN
    :param job_name: name of a job within the task graph
    For DNNs job_name = name of a DNN layer
    :return: task id (int)
    """
    for i in range(len(task_graph.tasks)):
        jobs = task_graph.jobs_per_task[i]
        if job_name in jobs:
            return i
    raise Exception("Mapping parsing error: job " + job_name +
                    " not found in the input task graph ")