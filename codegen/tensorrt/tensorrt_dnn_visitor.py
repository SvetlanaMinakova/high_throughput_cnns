from codegen.codegen_visitor import copy_static_app_code
from fileworkers.common_fw import create_or_overwrite_dir
from models.dnn_model.dnn import DNN
import codegen.tensorrt.cpp.cpp_dnn_visitor
import codegen.tensorrt.h.h_dnn_visitor
import codegen.makefile_generator
import codegen.app_main_generator
import codegen.codegen_config
from models.app_model.dnn_inf_model import generate_external_input_buffers, generate_external_output_buffers
from models.app_model.InterDNNConnection import InterDNNConnection


def visit_dnn(dnn: DNN, code_dir, verbose=True):
    """
    Generate TensorRT code for a DNN
    :param dnn: DNN to generate code for
    :param code_dir: folder to generate code in
    NOTE: folder will be overwritten!
    :param verbose: print details
    """
    create_or_overwrite_dir(code_dir)

    # attributes
    codegen_flags = [codegen.codegen_config.CodegenFlag.GPU_PROFILE]
    cpu_cores_allocation = {dnn.name: 1}
    class_names_in_exec_order = [dnn.name]
    gpu_partition_class_names = [dnn.name]
    cpu_partition_class_names = []

    # visit dnn
    codegen.tensorrt.cpp.cpp_dnn_visitor.visit_dnn(dnn, code_dir, gpu_profile=True)
    codegen.tensorrt.h.h_dnn_visitor.visit_dnn(dnn, code_dir, gpu_profile=True)

    # generate I/O buffers
    # No inter-dnn buffers are used for a single-dnn application
    io_buffers = []
    """
    input_buffers = generate_external_input_buffers(dnn)
    output_buffers = generate_external_output_buffers(dnn)
    io_buffers = input_buffers + output_buffers
    """

    # there is only one partition, and thus no connections between partitions
    inter_partition_connections = []

    # generate app main
    codegen.app_main_generator.generate_app_main(code_dir,
                                                 class_names_in_exec_order,
                                                 gpu_partition_class_names,
                                                 cpu_partition_class_names,
                                                 codegen_flags,
                                                 cpu_cores_allocation,
                                                 inter_partition_connections,
                                                 io_buffers)

    codegen.makefile_generator.generate_makefile(code_dir,
                                                 gpu_partition_class_names,
                                                 cpu_partition_class_names,
                                                 arm_cl=False, trt=True)

    copy_static_app_code(code_dir)
    if verbose:
        print("TensorRT code is generated in", code_dir)


def visit_dnn_partitioned(dnn_partitions: [DNN],
                          inter_partition_connections: [InterDNNConnection],
                          code_dir,
                          verbose=True):
    """
    Generate TensorRT code for a DNN
    :param dnn_partitions: list of (partitioned) DNNs
    :param inter_partition_connections: list of connections between DNN partitions
    :param code_dir: folder to generate code in
    NOTE: folder will be overwritten!
    :param verbose: print details
    """
    create_or_overwrite_dir(code_dir)
    codegen_flags = [codegen.codegen_config.CodegenFlag.GPU_PROFILE]

    # attributes
    class_names_in_exec_order = [partition.name for partition in dnn_partitions]
    gpu_partition_class_names = class_names_in_exec_order
    cpu_partition_class_names = []
    cpu_cores_allocation = {partition.name: 1 for partition in dnn_partitions}

    # visit every partition
    for partition in dnn_partitions:
        codegen.tensorrt.cpp.cpp_dnn_visitor.visit_dnn(partition, code_dir, gpu_profile=True)
        codegen.tensorrt.h.h_dnn_visitor.visit_dnn(partition, code_dir, gpu_profile=True)

    # generate app main
    codegen.app_main_generator.generate_app_main(code_dir,
                                                 class_names_in_exec_order,
                                                 gpu_partition_class_names,
                                                 cpu_partition_class_names,
                                                 codegen_flags,
                                                 cpu_cores_allocation,
                                                 inter_partition_connections)
    # generate makefile
    codegen.makefile_generator.generate_makefile(code_dir,
                                                 gpu_partition_class_names,
                                                 cpu_partition_class_names,
                                                 arm_cl=False, trt=True)

    copy_static_app_code(code_dir)
    if verbose:
        print("TensorRT code is generated in", code_dir)




