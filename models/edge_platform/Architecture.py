class Architecture:
    """
    Target edge hardware platform (architecture) class
    :param processors: list of unique processors names, e.g., ["CPU0", "CPU1", "CPU2", "CPU3", "CPU4", "GPU"]
    :param processors_types: list of processor types, e.g., ["large_CPU", "large_CPU", "small_CPU", "small_CPU", "GPU"]
        required: len(processors_types) = len(processors)
    :param processor_types_distinct: list of distinct processor types, e.g., ["large_CPU", "small_CPU", "GPU"]
    """
    def __init__(self, processors, processors_types,  processor_types_distinct):
        self.name = "architecture"
        self.processors = processors
        self.processors_types = processors_types
        self.processors_num = len(processors)
        self.processors_types_distinct = processor_types_distinct
        self.processor_types_distinct_num = len(processor_types_distinct)
        # names of processors that are accelerators
        self.accelerators = []
        # Deprecated (replaced by communication matrix)
        # self.communication_channels = []
        # communication speed (in MegaBytes per second) between every pair of processors
        # available on the platform
        self.communication_speed_matrix_mb_s = [[0 for i in range(self.processors_num)] for j in range(self.processors_num)]
        # max flops for every type of processor: needed for flop-based evaluation
        self.max_giga_flops_per_proc_type = [1 for _ in processor_types_distinct]

    def set_communication_speed_mb_s(self, src_proc_id, dst_proc_id, communication_speed_mb_s):
        self.communication_speed_matrix_mb_s[src_proc_id][dst_proc_id] = communication_speed_mb_s

    def get_communication_speed_mb_s(self, src_proc_id, dst_proc_id):
        return self.communication_speed_matrix_mb_s[src_proc_id][dst_proc_id]

    def get_proc_type_id(self, processor_id):
        """
        Get id of distinct processor type by processor id
        :param processor_id: processor id
        :return: id of distinct processor type
        """
        processor_type = self.processors_types[processor_id]
        for i in range(len(self.processors_types_distinct)):
            if processor_type == self.processors_types_distinct[i]:
                return i
        return None

    def get_max_giga_flops_for_proc_type(self, proc_type_id):
        """
        Get max flops for processor
        :param proc_type_id: distinct processor type id
        :return: max flops for processor
        """
        if len(self.max_giga_flops_per_proc_type) > proc_type_id:
            return self.max_giga_flops_per_proc_type[proc_type_id]
        return 1

    def get_proc_id_by_name(self, name):
        for proc_id in range(len(self.processors)):
            if self.processors[proc_id] == name:
                return proc_id

    def get_first_accelerator_proc_id(self):
        if self.accelerators:
            accelerator_name = self.accelerators[0]
            first_accelerator_id = self.get_proc_id_by_name(accelerator_name)
            return first_accelerator_id
        return -1


def get_jetson():
    """
    Get Jetson as architecture example
    """
    # baseline architecture
    processors = ["CPU0", "CPU1", "CPU2", "CPU3", "CPU4", "GPU"]
    processor_types = ["large_CPU", "large_CPU", "small_CPU", "small_CPU", "small_CPU", "GPU"]
    processor_types_distinct = ["large_CPU", "small_CPU", "GPU"]
    jetson = Architecture(processors, processor_types, processor_types_distinct)
    jetson.name = "Jetson"

    # bandwidth between processors
    # CPU/GPU bandwidth = 20 GB/s
    cpu_gpu_bandwidth_mb_s = 20 * 1e9/1e6
    for src_processor_id in range(len(jetson.processors)):
        for dst_processor_id in range(len(jetson.processors)):
            src_processor_type = jetson.processors_types[src_processor_id]
            dst_processor_type = jetson.processors_types[dst_processor_id]
            if src_processor_type in ["large_CPU", "small_CPU"] and dst_processor_type == "GPU" or\
                    src_processor_type == "GPU" and dst_processor_type in ["large_CPU", "small_CPU"]:
                jetson.set_communication_speed_mb_s(src_processor_id, dst_processor_id, cpu_gpu_bandwidth_mb_s)

    # list of accelerators
    jetson.accelerators.append("GPU")

    # max flops for every type of processor: needed for flop-based throughput evaluation
    # FLOPS are computed using NVIDIA docs (for GPU) and https://en.wikipedia.org/wiki/FLOPS for CPU
    gpu_glops = 250 # 667: 667 is max, but GPU is never fully occupied due to registers limitation
    large_cpu_gflops = 16.28
    small_cpu_gflops = 10.85
    jetson.max_giga_flops_per_proc_type = [large_cpu_gflops, small_cpu_gflops, gpu_glops]

    return jetson

