from codegen.codegen_visitor import copy_static_app_code
from fileworkers.common_fw import create_or_overwrite_dir
from models.dnn_model.dnn import DNN
import codegen.arm_cl.cpp.cpp_dnn_visitor
import codegen.arm_cl.h.h_dnn_visitor
import codegen.makefile_generator
import codegen.app_main_generator
import codegen.codegen_config
from codegen.arm_cl.dnn_to_streams import DNNSubStreamsGenerator
from DSE.buffers_generation.inter_dnn_buffers_builder import generate_inter_partition_buffers
from models.app_model.InterDNNConnection import InterDNNConnection
from DSE.scheduling.dnn_scheduling import DNNScheduling


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
    class_names_in_exec_order = [dnn.name]
    gpu_partition_class_names = []
    cpu_partition_class_names = [dnn.name]
    cpu_cores_allocation = {dnn.name: 1}
    codegen_flags = [codegen.codegen_config.CodegenFlag.CPU_PROFILE]

    # represent dnn as a set of ARM-CL sub-streams
    # where every parallel branch or residual connection
    # is defined in a separate stream
    sub_streams_generator = DNNSubStreamsGenerator(dnn)
    sub_streams_generator.dnn_to_sub_streams()
    # visit DNN
    codegen.arm_cl.cpp.cpp_dnn_visitor.visit_dnn(dnn, code_dir, profile=True,
                                                 sub_streams_generator=sub_streams_generator)
    codegen.arm_cl.h.h_dnn_visitor.visit_dnn(dnn, code_dir, profile=True,
                                             sub_streams_generator=sub_streams_generator)

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

    # generate makefile
    codegen.makefile_generator.generate_makefile(code_dir,
                                                 gpu_partition_class_names,
                                                 cpu_partition_class_names,
                                                 arm_cl=True,
                                                 trt=False)
    copy_static_app_code(code_dir, verbose=verbose)
    if verbose:
        print("ARM-CL (CPU) code is generated in", code_dir)


def visit_dnn_partitioned(dnn_partitions: [DNN],
                          inter_partition_connections: [InterDNNConnection],
                          code_dir,
                          verbose=True):
    """
    Generate ARM-CL code for a DNN
    :param dnn_partitions: list of (partitioned) DNNs
    :param inter_partition_connections: list of connections between DNN partitions
    :param code_dir: folder to generate code in
    NOTE: folder will be overwritten!
    :param verbose: print details
    """
    create_or_overwrite_dir(code_dir)
    codegen_flags = [codegen.codegen_config.CodegenFlag.CPU_PROFILE]

    # attributes
    class_names_in_exec_order = [partition.name for partition in dnn_partitions]
    gpu_partition_class_names = []
    cpu_partition_class_names = class_names_in_exec_order
    cpu_cores_allocation = {partition.name: 1 for partition in dnn_partitions}

    # visit every partition
    for partition in dnn_partitions:
        # represent dnn as a set of ARM-CL sub-streams
        # where every parallel branch or residual connection
        # is defined in a separate stream
        sub_streams_generator = DNNSubStreamsGenerator(partition)
        sub_streams_generator.dnn_to_sub_streams()
        # visit DNN
        codegen.arm_cl.cpp.cpp_dnn_visitor.visit_dnn(partition, code_dir, profile=True,
                                                     sub_streams_generator=sub_streams_generator)
        codegen.arm_cl.h.h_dnn_visitor.visit_dnn(partition, code_dir, profile=True,
                                                 sub_streams_generator=sub_streams_generator)
    # generate I/O buffers
    io_buffers = generate_inter_partition_buffers(inter_partition_connections, schedule_type=DNNScheduling.SEQUENTIAL)

    # generate app main
    codegen.app_main_generator.generate_app_main(code_dir,
                                                 class_names_in_exec_order,
                                                 gpu_partition_class_names,
                                                 cpu_partition_class_names,
                                                 codegen_flags,
                                                 cpu_cores_allocation,
                                                 inter_partition_connections,
                                                 io_buffers
                                                 )
    # generate makefile
    codegen.makefile_generator.generate_makefile(code_dir,
                                                 gpu_partition_class_names,
                                                 cpu_partition_class_names,
                                                 arm_cl=True, trt=False)

    copy_static_app_code(code_dir, verbose=verbose)
    if verbose:
        print("ARM-CL (CPU) partitioned code is generated in", code_dir)




