from codegen.codegen_visitor import create_or_overwrite_code_dir, copy_static_app_code
from models.dnn_model.dnn import DNN
import codegen.arm_cl.cpp.cpp_dnn_visitor
import codegen.arm_cl.h.h_dnn_visitor
import codegen.makefile_generator
import codegen.app_main_generator
import codegen.codegen_config
from codegen.arm_cl.dnn_to_streams import DNNSubStreamsGenerator


def visit_dnn(dnn: DNN, code_dir, verbose=True):
    """
    Generate TensorRT code for a DNN
    :param dnn: DNN to generate code for
    :param code_dir: folder to generate code in
    NOTE: folder will be overwritten!
    :param verbose: print details
    """
    create_or_overwrite_code_dir(code_dir)

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

    # generate app main
    codegen.app_main_generator.generate_app_main(code_dir,
                                                 class_names_in_exec_order,
                                                 gpu_partition_class_names,
                                                 cpu_partition_class_names,
                                                 codegen_flags,
                                                 cpu_cores_allocation)

    # generate makefile
    codegen.makefile_generator.generate_makefile(code_dir,
                                                 gpu_partition_class_names,
                                                 cpu_partition_class_names,
                                                 arm_cl=True,
                                                 trt=False)
    copy_static_app_code(code_dir)
    if verbose:
        print("ARM-CL code is generated in", code_dir)


def visit_dnn_partitioned(dnn_partitions: [DNN], code_dir, verbose=True):
    """
    Generate TensorRT code for a DNN
    :param dnn_partitions: list of (partitioned) DNNs
    :param code_dir: folder to generate code in
    NOTE: folder will be overwritten!
    :param verbose: print details
    """
    create_or_overwrite_code_dir(code_dir)
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

    # generate app main
    codegen.app_main_generator.generate_app_main(code_dir,
                                                 class_names_in_exec_order,
                                                 gpu_partition_class_names,
                                                 cpu_partition_class_names,
                                                 codegen_flags,
                                                 cpu_cores_allocation)
    # generate makefile
    codegen.makefile_generator.generate_makefile(code_dir,
                                                 gpu_partition_class_names,
                                                 cpu_partition_class_names,
                                                 arm_cl=True, trt=False)

    copy_static_app_code(code_dir)
    if verbose:
        print("TensorRT code is generated in", code_dir)




