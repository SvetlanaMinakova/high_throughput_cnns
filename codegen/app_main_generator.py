from codegen.codegen_visitor import CodegenVisitor
from codegen.codegen_config import CodegenFlag
from codegen.buffers_visitor import buf_type_to_buf_class
from models.app_model.InterDNNConnection import InterDNNConnection
from models.data_buffers.data_buffers import DataBuffer


def generate_app_main(directory,
                      class_names_in_exec_order: [],
                      gpu_partition_class_names: [],
                      cpu_partition_class_names: [],
                      flags: [CodegenFlag],
                      cpu_core_per_class_name: {},
                      inter_partition_connections: [InterDNNConnection],
                      inter_partition_buffers: [DataBuffer]):
    """
        generate main application file
        :param directory: directory to generate main application file in
        :param class_names_in_exec_order: CPU/GPU partition class names in execution order
        :param gpu_partition_class_names: names of GPU (TensorRT (sub) DNNs)
        :param cpu_partition_class_names: names of CPU (ARM-CL (sub) DNNs)
        :param cpu_core_per_class_name: (dictionary) key (string)=class_name, value(int) = cpu_core_id
         allocation of CPU cores for every partition
        :param flags: code-generation flags (see  codegen_config.CodegenFlags)
        :param inter_partition_connections: list of connections between dnn partitions
        :param inter_partition_buffers: (optional) list of buffers, used to store data
            exchanged between DNN partitions
    """
    filepath = directory + "/" + "appMain.cpp"
    with open(filepath, "w") as print_file:
        visitor = AppMainGenerator(print_file,
                                   class_names_in_exec_order,
                                   gpu_partition_class_names,
                                   cpu_partition_class_names,
                                   flags,
                                   cpu_core_per_class_name,
                                   inter_partition_connections,
                                   inter_partition_buffers)
        visitor.visit()


class AppMainGenerator(CodegenVisitor):
    """
     * Generator of main application file, which contains
     * application structure and high-level control logic
    """
    def __init__(self, print_file,
                 class_names_in_exec_order: [],
                 gpu_partition_class_names: [],
                 cpu_partition_class_names: [],
                 flags: [CodegenFlag],
                 cpu_core_per_class_name: {},
                 inter_partition_connections: [],
                 inter_partition_buffers: []):
        """
        Create new application main-file generator
        :param print_file: file to print app-main code in
        :param gpu_partition_class_names: names of GPU (TensorRT (sub) DNNs)
        :param cpu_partition_class_names: names of CPU (ARM-CL (sub) DNNs)
        :param flags: code-generation flags (see  codegen_config.CodegenFlags)
        :param cpu_core_per_class_name: (dictionary) key (string)=class_name, value(int) = cpu_core_id
         allocation of CPU cores for every partition
        :param inter_partition_connections: list of connections between dnn partitions
        :param inter_partition_buffers: (optional) list of buffers, used to store data
            exchanged between DNN partitions
        """
        super().__init__(print_file, prefix="")
        self.class_names_in_exec_order = class_names_in_exec_order
        self.gpu_partition_class_names = gpu_partition_class_names
        self.cpu_partition_class_names = cpu_partition_class_names
        self.arm_cl = True if len(cpu_partition_class_names) > 0 else False
        self.trt = True if len(gpu_partition_class_names) > 0 else False
        self.flags = flags
        self.cpu_core_per_class_name = cpu_core_per_class_name
        self.default_cpu_core = 1
        self.inter_partition_connections = inter_partition_connections
        self.inter_partition_buffers = inter_partition_buffers

        # meta-data
        # names of partitions and engines
        self.gpu_partition_names = []
        self.gpu_engine_names = []
        self.cpu_partition_names = []
        self.cpu_engine_names = []
        self.name_partitions_and_engines()

    def name_partitions_and_engines(self):
        """
        Give names to CPU/GPU partitions and engines
        """
        partition_id = 0
        for class_name in self.class_names_in_exec_order:
            added = False
            partition_name = "p" + str(partition_id)
            engine_name = "e" + str(partition_id)
            if class_name in self.gpu_partition_class_names:
                self.gpu_partition_names.append(partition_name)
                self.gpu_engine_names.append(engine_name)
                added = True
            if class_name in self.cpu_partition_class_names:
                self.cpu_partition_names.append(partition_name)
                self.cpu_engine_names.append(engine_name)
                added = True
            if added:
                partition_id += 1
            else:
                print("WARNING: AppMain creation: unspecified mapping of class" + class_name)

    def visit(self):
        self._write_includes()
        self.write_line("")
        self.write_line("using namespace std;")
        self.write_line("")
        self._write_main()

    def _write_includes(self):
        """
        Write includes in the application main class .cpp beginning
        """
        std_lib_headers = ["iostream", "map", "vector", "thread", "chrono"]
        gpu_lib_headers = ["cuda_runtime_api", "gpu_partition", "gpu_engine"]
        cpu_lib_headers = ["arm_compute/graph", "cpu_engine"]
        custom_buffer_headers = ["types", "SingleBuffer", "DoubleBuffer", "Timer"] # "SharedBuffer"

        local_lib_headers = []
        if self.trt > 0:
            local_lib_headers.extend(gpu_lib_headers)
            local_lib_headers.extend(self.gpu_partition_class_names)
        if self.arm_cl > 0:
            local_lib_headers.extend(cpu_lib_headers)
            local_lib_headers.extend(self.cpu_partition_class_names)

        for lib_header in std_lib_headers:
            self._include_std_cpp_header(lib_header)

        for lib_header in local_lib_headers:
            self._include_local_cpp_header(lib_header)

        if self.inter_partition_buffers is not None:
            for lib_header in custom_buffer_headers:
                self._include_local_cpp_header(lib_header)

    def _write_main(self):
        self.write_line("int main (int argc, char **argv) {")
        self.prefix_inc()
        if self.trt:
            self.write_line("cudaDeviceReset();")

        self.write_line("std::cout<<\"***DNN building phase.***\"<<std::endl;")
        self._create_partitions()
        self._create_engines()
        self._create_buffers()
        self._create_pthread_parameters()
        self._inference()
        self._clean_memory()
        self.write_line("")
        self.write_line("return 0;")
        self.prefix_dec()
        self.write_line("}")

    def _create_partitions(self):
        self.write_line("////////////////////////////////////////////////////////////")
        self.write_line("// CREATE PARTITIONS (DNN-DEPENDENT TOPOLOGY INIT/CLEAN) //")
        self.write_line("std::cout<<\" - partitions creation.\"<<std::endl;")
        self.write_line("//GPU")
        for pid in range(len(self.gpu_partition_class_names)):
            partition_class = self.gpu_partition_class_names[pid]
            partition_name = self.gpu_partition_names[pid]
            self.write_line(partition_class + " " + partition_name + ";")
        self.write_line("")
        self.write_line("//CPU")
        for pid in range(len(self.cpu_partition_class_names)):
            partition_class = self.cpu_partition_class_names[pid]
            partition_name = self.cpu_partition_names[pid]
            self.write_line(partition_class + " " + partition_name + ";")
        self.write_line("")

    ##############
    # BUFFERS
        
    def _create_buffers(self):
        self.write_line("////////////////////////////////////////////")
        self.write_line("// CREATE BUFFERS BETWEEN DNN PARTITIONS //")
        self.write_line("std::cout<<\" - I/O buffers allocation.\"<<std::endl;")
        if len(self.inter_partition_buffers) > 0:
            self._define_inter_partition_buffers()
            self.write_line("")
            self._allocate_inter_partition_buffers()
        self.write_line("")

    ####################################
    #  buffers between DNN partitions
    def _define_inter_partition_buffers(self):
        self.write_line("// create and init buffers")
        for buffer in self.inter_partition_buffers:
            buf_class = buf_type_to_buf_class(buffer.type)
            self.write_line(buf_class + " " + buffer.name + ";")
            self.write_line(buffer.name + ".init(" + "\"" + buffer.name + "\"" + ", " + str(buffer.size) + ");")

    def _allocate_inter_partition_buffers(self):
        self.write_line("// allocate buffers to cpu/gpu engines")
        for buffer in self.inter_partition_buffers:
            self.write_line("// " + buffer.name)
            source_class_names = self._get_buf_src_class_names(buffer)
            destination_class_names = self._get_buf_dst_class_names(buffer)

            for class_name in source_class_names:
                engine_name = self._get_engine_name(class_name)
                self.write_line(engine_name + ".addOutputBufferPtr(&" + buffer.name + ");")

            for class_name in destination_class_names:
                engine_name = self._get_engine_name(class_name)
                self.write_line(engine_name + ".addInputBufferPtr(&" + buffer.name + ");")

    def _get_buf_src_class_names(self, buffer):
        if buffer.subtype == "input_buffer":
            return buffer.users
        if buffer.subtype == "output_buffer":
            return []
        # i/o buffer
        # extract i/o buffer sources from connection
        src_class_names = []
        for connection in self.inter_partition_connections:
            if connection.name in buffer.users:
                src_class_names.append(connection.src.name)
        return src_class_names

    def _get_buf_dst_class_names(self, buffer):
        if buffer.subtype == "input_buffer":
            return []
        if buffer.subtype == "output_buffer":
            return buffer.users
        # i/o buffer
        # extract i/o buffer sources from connection
        dst_class_names = []
        for connection in self.inter_partition_connections:
            if connection.name in buffer.users:
                dst_class_names.append(connection.dst.name)
        return dst_class_names

    def _get_engine_name(self, class_name: str):
        engine_id = self._get_class_id(class_name)
        engine_name = "e" + str(engine_id)
        return engine_name

    def _get_partition_name(self, class_name: str):
        partition_id = self._get_class_id(class_name)
        partition_name = "p" + str(partition_id)
        return partition_name

    def _get_class_id(self, class_name: str):
        partition_id = 0
        for name in self.class_names_in_exec_order:
            if name == class_name:
                return partition_id
            partition_id += 1
        raise Exception("Class name " + class_name + " not defined in the partition classes list")

    #########################
    ### engines

    def _create_engines(self):
        self.write_line("// CREATE ENGINES (OBJECTS TO RUN DNN PARTITIONS) //")
        self.write_line("std::cout<<\" - Engines creation.\"<<std::endl;")
        self.write_line("//GPU")
        for pid in range(len(self.gpu_partition_names)):
            partition_name = self.gpu_partition_names[pid]
            engine_name = self.gpu_engine_names[pid]
            self.write_line("cudaStream_t " + partition_name + "_stream; ")
            self.write_line("CHECK(cudaStreamCreate(&" + partition_name + "_stream));")
            # I/O s moved to SharedBuffers
            self.write_line("gpu_engine " + self.gpu_engine_names[pid] + " (&" + partition_name +
                            ", &" + partition_name + "_stream, \"" + engine_name + "\");")
            """
            self.write_line("gpu_engine " + self.gpu_engine_names[pid] + " (&" + partition_name + ", " +
                            partition_name + "_input, " + partition_name + "_output, &" + 
                            partition_name + "_stream, \"" + engine_name + "\");")
            """
        self.write_line("")
        self.write_line("//CPU")
        for pid in range(len(self.cpu_engine_names)):
            engine_name = self.cpu_engine_names[pid]
            partition_name = self.cpu_partition_names[pid]
            # I/O s moved to SharedBuffers
            self.write_line("cpu_engine " + engine_name + " (argc, argv, &" +
                            partition_name + ", \"" + engine_name + "\");")
            """
            self.write_line("cpu_engine " + engine_name + " (argc, argv, " +
                            partition_name + "_input, " + partition_name + "_output, &" +
                            partition_name + ", \"" + engine_name + "\");")
            """

        # if (_cpuDebugMode)
        if len(self.cpu_partition_names) > 0:
            self.write_line("//CPU engine pointers")
            self.write_line("std::vector<cpu_engine*> cpu_engine_ptrs;")
            for cpu_engine_name in self.cpu_engine_names:
                self.write_line("cpu_engine_ptrs.push_back(&" + cpu_engine_name + ");")
            self.write_line("")

        # if (_gpuDebugMode) {
        if len(self.gpu_partition_names) > 0:
            self.write_line("//GPU engine pointers")
            self.write_line("std::vector<gpu_engine*> gpu_engine_ptrs;")
            for gpu_engine_name in self.gpu_engine_names:
                self.write_line("gpu_engine_ptrs.push_back(&" + gpu_engine_name + ");")
            self.write_line("")
            
    def _create_pthread_parameters(self):
        self.write_line("/////////////////////////////////////////////////////////////")
        self.write_line("//  PTHREAD PARAMETERS //")
        self.write_line("std::cout<<\" - Pthread info-params creation.\"<<std::endl;")
        self.write_line("")
        self.write_line("int subnets = " + str(len(self.class_names_in_exec_order)) + ";")

        core_ids_per_partition = "int core_ids[subnets] = {"
        for i in range(len(self.class_names_in_exec_order)-1):
            partition_class_name = self.class_names_in_exec_order[i]
            cpu_core_id = self._get_cpu_core_id(partition_class_name)
            core_ids_per_partition += str(cpu_core_id) + ", "

        partition_class_name = self.class_names_in_exec_order[-1]
        cpu_core_id = self._get_cpu_core_id(partition_class_name)
        core_ids_per_partition += str(cpu_core_id) + "};"
        self.write_line(core_ids_per_partition)

        self.write_line("//Allocate memory for pthread_create() arguments")
        self.write_line("const int num_threads = subnets;")
        self.write_line("struct ThreadInfo* thread_info = "
                        "(struct ThreadInfo*)(calloc(num_threads, sizeof(struct ThreadInfo)));")
        self.write_line("")

        self.write_line("//  allocate CPU cores")
        self.write_line("for(int i = 0;i<num_threads; i++)")
        self.prefix_inc()
        self.write_line("thread_info[i].core_id = core_ids[i];")
        self.prefix_dec()
        self.write_line("")

    def _get_cpu_core_id(self, partition_class_name):
        if partition_class_name in self.cpu_core_per_class_name.keys():
            return self.cpu_core_per_class_name[partition_class_name]
        return self.default_cpu_core

    def _inference(self):
        self.write_line("/////////////////////////////////////////////////////////////")
        self.write_line("// INFERENCE //")
        self.write_line("std::cout<<\"*** DNN inference phase.***\"<<std::endl;")
        self.write_line("")
        if CodegenFlag.CPU_PROFILE in self.flags:
            self._cpu_profile_inference()
        else:
            self._combined_inference()
        self.write_line("")
    
    def _combined_inference(self):
        self.write_line("std::cout<<\" - Threads creation and execution.\"<<std::endl;")
        self.write_line("")
        self.write_line("auto startTime = std::chrono::high_resolution_clock::now();")
        self.write_line("")
        self.write_line("//Create and run posix threads")
        thread_id = 0
        for class_name in self.class_names_in_exec_order:
            engine_name = "e" + str(thread_id)
            if class_name in self.gpu_partition_class_names:
                self.write_line("std::thread my_thread" + str(thread_id) +
                                "(&gpu_engine::main, &" + engine_name +
                                ", &thread_info[" + str(thread_id) + "]);//(GPU)")
            else:
                self.write_line("std::thread my_thread" + str(thread_id) +
                                "(&cpu_engine::main, &" + engine_name +
                                ", &thread_info[" + str(thread_id) + "]);//(CPU)")
            thread_id += 1
        self.write_line("")

        self.write_line("//join posix threads")
        for i in range(len(self.class_names_in_exec_order)):
            self.write_line("my_thread" + str(i) + ".join();")
        self.write_line("")

        self.write_line("auto endTime = std::chrono::high_resolution_clock::now();")
        self.write_line("float totalTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();")
        self.write_line("std::cout<<\"Average over \"<<frames<< \" images = "
                        "~ \"<<(totalTime/float(frames))<<\" ms/img\"<<std::endl;")

    def _cpu_profile_inference(self):
        """
        CPU-execution in profile (per-layer) mode
        """
        self.write_line("// set CPU ids here")
        self.write_line("int large_cpu_id = 1;")
        self.write_line("int small_cpu_id = 4;")

        self.write_line("")
        self.write_line("std::cout<<\"CPU eval_table over \"<<frames<< \" images\"<<std::endl;")
        self.write_line("std::vector<float> cpu_time;")
        # run everything on large CPU
        self._cpu_debug_inference("large_cpu_id", "large_CPU")

        self.write_line("// clear time direct_measurements")
        self.write_line("cpu_time.clear();")
        self.write_line("")
        # run everything on small CPU
        self._cpu_debug_inference("small_cpu_id", "small_CPU")
        self.write_line("")

    def _cpu_debug_inference(self, core_id_str, benchmark_label: str):
        self.write_line("// allocate CPU cores")
        self.write_line("for(int i=0; i<num_threads; i++)")
        self.prefix_inc()
        self.write_line("thread_info[i].core_id = " + core_id_str + ";")
        self.prefix_dec()
        self.write_line("")
        self.write_line("//run eval_table")
        self.write_line("for(int en=0; en<cpu_engine_ptrs.size();en++) {")
        self.prefix_inc()
        self.write_line("")
        self.write_line("auto startTime = std::chrono::high_resolution_clock::now();")
        self.write_line("")

        self.write_line("std::thread my_thread(&cpu_engine::main, cpu_engine_ptrs.at(en), &thread_info[en]);//(CPU)")
        self.write_line("my_thread.join();")
        self.write_line("auto endTime = std::chrono::high_resolution_clock::now();")
        self.write_line("float totalTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();")
        self.write_line("cpu_time.push_back((totalTime/float(frames)));")
        self.prefix_dec()
        self.write_line("}")

        self.write_line("")
        self.write_line("std::cout<<\"\\\"" + benchmark_label + "\\\" : [\";")
        self.write_line("for(int i=0; i< cpu_time.size() - 1; i++)")
        self.prefix_inc()
        self.write_line("std::cout<<cpu_time.at(i)<<\", \";")
        self.prefix_dec()
        self.write_line("std::cout<<(cpu_time.at(cpu_time.size() - 1))<<\"], \"<<std::endl;")
        self.write_line("")

    def _clean_memory(self):
        """Clean platform memory"""
        self.write_line("/////////////////////////////////////////////////////////////")
        self.write_line("// CLEAN MEMORY //")
        self.write_line("std::cout<<\"*** DNN destruction phase ***\"<<std::endl;")
        self.write_line("")
        self.write_line("//Destroy GPU streams")
        self.write_line("std::cout<<\" - CUDA streams destruction\"<<std::endl;")
        for gpu_partition in self.gpu_partition_names:
            self.write_line("cudaStreamDestroy(" + gpu_partition + "_stream);")
        self.write_line("")

        self.write_line("//Destroy CPU partitions")
        self.write_line("std::cout<<\" - CPU partitions destruction\"<<std::endl;")
        for cpu_partition in self.cpu_partition_names:
            self.write_line(cpu_partition + ".do_teardown();")

        self.write_line("")
        self.write_line("//delete pthread parameters")
        self.write_line("std::cout<<\" - Pthread parameters destruction\"<<std::endl;")

        self.write_line("free(thread_info);")
        self.write_line("")

