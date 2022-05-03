from codegen.codegen_visitor import copy_static_app_code
from fileworkers.common_fw import create_or_overwrite_dir
from models.dnn_model.dnn import DNN
from models.app_model.dnn_inf_model import DNNInferenceModel
from DSE.partitioning.after_mapping.partition_dnn_with_inf_model import partition_dnn_with_dnn_inference_model
from codegen.mixed.mixed_dnn_visitor import get_gpu_partition_class_names, get_cpu_cores_allocation
# per-partition H and CPP files
import codegen.wrapper.cpp.cpp_dnn_visitor
import codegen.wrapper.h.h_dnn_visitor
# other (common) code
import codegen.wrapper.wrapper_app_main_generator
import codegen.wrapper.wrapper_makefile_generator
import codegen.codegen_config


def visit_dnn_app(dnn: DNN,
                  dnn_inf_model: DNNInferenceModel,
                  code_dir: str,
                  time_per_subnet: {},
                  verbose=False):
    """
    Generate ARM-CL code for a DNN
    :param dnn: deep neural network
    :param dnn_inf_model: DNN inference model that specifies
        partitioning, mapping and scheduling of the dnn on the target platform
    :param code_dir: folder to generate code in
    NOTE: folder will be overwritten!
    :param verbose: print details
    :param time_per_subnet dictionary where key (string) is the name of a
        sub-network, value (float) is the sub-network execution time. If a
        sub-network is not mentioned in the dictionary, its execution time is 0.
    """
    create_or_overwrite_dir(code_dir)
    codegen_flags = []

    # attributes
    class_names_in_exec_order = [partition["name"] for partition in dnn_inf_model.json_partitions]
    gpu_partition_class_names = get_gpu_partition_class_names(dnn_inf_model)
    # everything that is not executed on the GPU is executed on the CPU
    cpu_partition_class_names = [name for name in class_names_in_exec_order if name not in gpu_partition_class_names]
    cpu_cores_allocation = get_cpu_cores_allocation(dnn_inf_model)

    dnn_partitions, connections = partition_dnn_with_dnn_inference_model(dnn, dnn_inf_model)

    # visit every partition
    for partition in dnn_partitions:
        target_proc_type = "GPU" if partition.name in gpu_partition_class_names else "CPU"
        codegen.wrapper.cpp.cpp_dnn_visitor.visit_dnn(partition, code_dir, target_proc_type)
        subnet_time = get_subnet_exec_time(partition.name, time_per_subnet)
        codegen.wrapper.h.h_dnn_visitor.visit_dnn(partition, code_dir, target_proc_type, subnet_time)

    # generate app main
    codegen.wrapper.wrapper_app_main_generator.generate_app_main(code_dir,
                                                                 class_names_in_exec_order,
                                                                 gpu_partition_class_names,
                                                                 cpu_partition_class_names,
                                                                 cpu_cores_allocation,
                                                                 dnn_inf_model.json_connections,
                                                                 dnn_inf_model.inter_partition_buffers)

    # generate makefile
    codegen.wrapper.wrapper_makefile_generator.generate_makefile(code_dir, class_names_in_exec_order)
    # copy static code files
    static_code_path = "codegen/static_lib_files/wrapper"
    copy_static_app_code(code_dir, static_code_path, verbose)
    if verbose:
        print("Code wrapper is generated in", code_dir)


def get_subnet_exec_time(name, time_per_subnet: {}) -> int:
    """
    Get execution time of a dnn sub-network
    :param name: sub-network name
    :param time_per_subnet: dictionary where key (string) is the name of a
        sub-network, value (float) is the sub-network execution time. If a
        sub-network is not mentioned in the dictionary, its execution time is 0.
    :return: sub-network execution time (in milliseconds)
    """
    subnet_time = 0
    if name in time_per_subnet:
        subnet_time = int(time_per_subnet[name])
    return subnet_time


