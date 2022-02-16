from enum import Enum


def get_config():
    """
    CONFIG for makefile generation
    :return: CONFIG for makefile generation
    """
    # NOTE: Change this to specify paths to CUDA/ARM-CL libraries on your board
    cuda_path = "/usr/local/cuda-9.0"
    arm_cl_path = "/home/nvidia/arm_cl/ComputeLibrary"

    conf = {"cuda_path": cuda_path,
            "arm_cl_path": arm_cl_path}
    return conf


class CodegenFlag(Enum):
    # include CPU profiling
    CPU_PROFILE = 1
    # include GPU profiling
    GPU_PROFILE = 2
    # reuse buffers where possible
    REUSE_BUFFERS = 3

