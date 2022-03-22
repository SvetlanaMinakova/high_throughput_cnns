import os
import sys
import argparse
import traceback

# tests example
# python ./sdf_latency_eval_template.py -tg ./output/alexnet/task_graph.json -p ./output/architecture/jetson.json -o ./output/alexnet/

# ex 2
# ../kerasProj/venv/bin/python ./sdf_latency_eval_template.py --cnn /home/svetlana/ONNX/OnnxZooModels/mnist.onnx -tg ./output/mnist/task_graph.json -p ./output/architecture/jetson.json --flops -o ./output/mnist/


def main():
    # import current directory and it's subdirectories into system path for the current console
    # this would allow to import project modules without adding the project to the PYTHONPATH
    this_dir = os.path.dirname(__file__)
    sys.path.append(this_dir)

    # import project modules
    from util import print_stage
    from converters.json_converters.json_to_architecture import json_to_architecture
    from converters.json_converters.json_task_graph import parse_app_graph_json
    from eval.flops.layer_flops_estimator import eval_layer_flops
    from models.TaskGraph import get_task_id
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

    parser.add_argument('--cnn', metavar='--cnn', type=str, action='store', default=None,
                        help='path a CNN. Can be a path to: '
                             '1) a path to an .onnx file; '
                             '2) a path to .h5 file (cnn in format of Keras DL framework). '
                             'Only required if --flops flag is used')

    # general flags
    parser.add_argument("--silent", help="do not provide print-out for the script steps", action="store_true", default=False)
    parser.add_argument("--flops", help="use number of floating-point operations (FLOPS)"
                                        "to estimate execution time (latency) of a CNN."
                                        "If False, execution time will be set to 0 for every node.",
                        action="store_true", default=False)

    args = parser.parse_args()
    try:
        # Extract parameters from command-line arguments"
        task_graph_path = args.tg
        platform_path = args.p
        output_dir = args.o
        silent = args.silent
        verbose = not silent
        flops = args.flops
        cnn_path = args.cnn

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

        # eval latency using FLOPS
        if flops:
            from dnn_builders.input_dnn_manager import load_or_build_dnn_for_analysis
            stage = "Reading input DNN"
            print_stage(stage, verbose)
            if cnn_path is None:
                raise Exception("CNN has to be specified if --flops flag is used")
            dnn = load_or_build_dnn_for_analysis(cnn_path, verbose=verbose)
            for task_id in range(len(task_graph.tasks)):
                # compute task execution time (in FLOPS)
                task_time = 0
                jobs = task_graph.jobs_per_task[task_id]
                for job in jobs:
                    layer = dnn.find_layer_by_name(job)
                    if layer is not None:
                        layer_time = eval_layer_flops(layer)
                        task_time += layer_time
                # set task execution time (in FLOPS) for every processor
                for proc_type_distinct in architecture.processors_types_distinct:
                    template_as_dict[proc_type_distinct][task_id] = task_time

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

