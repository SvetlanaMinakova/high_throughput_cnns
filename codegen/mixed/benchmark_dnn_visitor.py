from models.dnn_model.dnn import DNN
import codegen.arm_cl.arm_cl_dnn_visitor
import codegen.tensorrt.tensorrt_dnn_visitor
from converters.dnn_to_task_graph import dnn_to_task_graph, dnn_to_task_graph_with_built_in
import sys
import traceback
from dnn_partitioning.before_mapping.partition_dnn_with_task_graph import partition_dnn_with_task_graph


def visit_dnn(dnn: DNN, code_dir, built_in, verbose=True):
    """
    Generate CPU/GPU code for direct measurements
    of DNN per-layer throughput/latency/energy on the platform
    :param dnn: DNN, represented as (analysis) DNN model
    :param code_dir: output code directory
    :param built_in: list of built-in operators
    :param verbose: print details
    """
    code_folder_gpu = code_dir + "gpu"
    code_folder_cpu = code_dir + "cpu"
    code_folder_cpu_partitioned = code_dir + "cpu_partitioned"
    stage = "beginning"
    try:
        stage = "GPU (whole) code generation"
        codegen.tensorrt.tensorrt_dnn_visitor.visit_dnn(dnn, code_folder_gpu, verbose)

        stage = "CPU (whole) code generation"
        codegen.arm_cl.arm_cl_dnn_visitor.visit_dnn(dnn, code_folder_cpu, verbose)

        stage = "task graph generation"
        if not built_in:
            tg = dnn_to_task_graph(dnn)
        else:
            tg = dnn_to_task_graph_with_built_in(dnn, built_in_ops=built_in)

        stage = "dnn dnn_partitioning with task graph"
        partitioned_dnn, connections = partition_dnn_with_task_graph(dnn, tg)

        stage = "CPU (partitioned) code generation"
        codegen.arm_cl.arm_cl_dnn_visitor.visit_dnn_partitioned(partitioned_dnn, code_folder_cpu_partitioned, verbose)

    except Exception as err:
        sys.stderr.write("Benchmark CPU/GPU code generation error. DNN: " + dnn.name +
                         " stage: " + stage + " reason: \n" + str(err))
        print(traceback.format_exc())

