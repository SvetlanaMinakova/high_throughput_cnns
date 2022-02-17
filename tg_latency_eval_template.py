import os
import sys
import argparse
import traceback

# test example
# python ./tg_latency_eval_template.py -tg ./output/alexnet/task_graph.json -p ./output/architecture/jetson.json -o ./output/alexnet/


def main():
    # import current directory and it's subdirectories into system path for the current console
    # this would allow to import project modules without adding the project to the PYTHONPATH
    this_dir = os.path.dirname(__file__)
    sys.path.append(this_dir)

    # import project modules
    from util import print_stage
    from converters.json_converters.json_to_architecture import json_to_architecture
    from converters.json_converters.json_task_graph import parse_app_graph_json
    from fileworkers.json_fw import save_as_json

    # general arguments
    parser = argparse.ArgumentParser(description='The script generates a JSON template for '
                                                 'per-node evaluation of SDF (task graph) model '
                                                 'execution time (latency) on the target edge platform.')

    parser.add_argument('-tg', metavar='--task-graph', type=str, action='store', required=True,
                        help='path to .json file with the task-graph (SDF) model generated '
                             'for the input cnn by the ./dnn_to_sdf_task_graph.py script')

    parser.add_argument('-p', metavar='--platform', type=str, action='store', required=True,
                        help='path to edge platform (architecture) description, saved in .json format')

    parser.add_argument('-o', metavar='--out-dir', type=str, action='store', default="./output",
                        help='path to output files directory')

    # general flags
    parser.add_argument("--silent", help="do not provide print-out for the script steps", action="store_true", default=False)

    args = parser.parse_args()
    try:
        # Extract parameters from command-line arguments"
        task_graph_path = args.tg
        platform_path = args.p
        output_dir = args.o
        silent = args.silent
        verbose = not silent
        # platform is not needed for this conversion

        stage = "Reading task graph (SDF) model "
        print_stage(stage, verbose)
        task_graph = parse_app_graph_json(task_graph_path)

        stage = "Reading target platform architecture"
        print_stage(stage, verbose)
        architecture = json_to_architecture(platform_path)

        stage = "Create template"
        print_stage(stage, verbose)

        # create tasks (layers) description
        tasks_description = tasks_description_to_json(task_graph)
        template_as_dict = {"layers": tasks_description}

        # create moc-up per-layer per-processor execution time (latency) estimation
        for proc_type_distinct in architecture.processors_types_distinct:
            latency_per_task = [0.0 for _ in range(len(task_graph.tasks))]
            template_as_dict[proc_type_distinct] = latency_per_task

        stage = "Save template as .json"
        print_stage(stage, verbose)
        eval_path = str(os.path.join(output_dir, "eval_template.json"))
        save_as_json(eval_path, template_as_dict)

    except Exception as e:
        print("Latency evaluation template creation error: " + str(e))
        traceback.print_tb(e.__traceback__)


def tasks_description_to_json(task_graph):
    """
    Get description of jobs (in the task graph)
    in JSON format
    :param task_graph: task graph
    :return:  description of jobs (in the task graph)
    in JSON format
    """
    tasks_descriptions = []
    for task_id in range(len(task_graph.tasks)):
        jobs = task_graph.jobs_per_task[task_id]
        task_description = ""
        if len(jobs) > 0:
            task_description = jobs[0]
            if len(jobs) > 1:
                for i in range(1, len(jobs)):
                    task_description = task_description + "_" + jobs[i]

        tasks_descriptions.append(task_description)
    return tasks_descriptions


if __name__ == "__main__":
    main()

