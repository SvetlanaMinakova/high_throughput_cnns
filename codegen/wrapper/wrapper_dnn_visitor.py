from codegen.codegen_visitor import copy_static_app_code
from fileworkers.common_fw import create_or_overwrite_dir
from models.dnn_model.dnn import DNN
from models.edge_platform.Architecture import Architecture
from models.app_model.dnn_inf_model import DNNInferenceModel
from dnn_partitioning.after_mapping.partition_dnn_with_inf_model import partition_dnn_with_dnn_inference_model
from codegen.mixed.mixed_dnn_visitor import get_gpu_partition_class_names, get_cpu_cores_allocation
"""
# tensorRT (GPU) code
import codegen.tensorrt.cpp.cpp_dnn_visitor
import codegen.tensorrt.h.h_dnn_visitor
# ARM-CL (CPU) code
import codegen.arm_cl.cpp.cpp_dnn_visitor
import codegen.arm_cl.h.h_dnn_visitor
from codegen.arm_cl.dnn_to_streams import DNNSubStreamsGenerator
# other (common) code
import codegen.makefile_generator
import codegen.app_main_generator
"""
import codegen.codegen_config


def visit_dnn_app(dnn: DNN,
                  architecture: Architecture,
                  dnn_inf_model: DNNInferenceModel,
                  code_dir: str,
                  verbose=False):
    """
    Generate ARM-CL code for a DNN
    :param architecture: target platform architecture
    :param dnn: deep neural network
    :param dnn_inf_model: DNN inference model that specifies
        partitioning, mapping and scheduling of the dnn on the target platform
    :param code_dir: folder to generate code in
    NOTE: folder will be overwritten!
    :param verbose: print details
    """
    create_or_overwrite_dir(code_dir)
    codegen_flags = []

    # attributes
    class_names_in_exec_order = [partition["name"] for partition in dnn_inf_model.partitions]
    gpu_partition_class_names = get_gpu_partition_class_names(dnn_inf_model)
    # everything that is not executed on the GPU is executed on the CPU
    cpu_partition_class_names = [name for name in class_names_in_exec_order if name not in gpu_partition_class_names]
    cpu_cores_allocation = get_cpu_cores_allocation(dnn_inf_model)

    dnn_partitions, connections = partition_dnn_with_dnn_inference_model(dnn, dnn_inf_model)

    """
    # visit every GPU partition
    for partition in dnn_partitions:
        if partition.name in gpu_partition_class_names:
            codegen.tensorrt.cpp.cpp_dnn_visitor.visit_dnn(partition, code_dir, gpu_profile=True)
            codegen.tensorrt.h.h_dnn_visitor.visit_dnn(partition, code_dir, gpu_profile=True)

    # visit every CPU partition
    for partition in dnn_partitions:
        if partition.name in cpu_partition_class_names:
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
                                                 arm_cl=True,
                                                 trt=True)
    """
    static_code_path = "codegen/static_lib_files/wrapper"
    copy_static_app_code(code_dir, static_code_path)
    if verbose:
        print("Code wrapper is generated in", code_dir)

