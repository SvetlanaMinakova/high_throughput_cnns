from codegen.codegen_visitor import CodegenVisitor
from models.data_buffers import DataBuffer
from codegen.wrapper.static_lib_info import get_static_lib_class_names
from codegen.buffers_visitor import buf_type_to_buf_class


def generate_app_main(directory,
                      class_names_in_exec_order: [],
                      gpu_partition_class_names: [],
                      cpu_partition_class_names: [],
                      cpu_core_per_class_name: {},
                      inter_partition_connections_desc: [],
                      inter_partition_buffers):
    """
        generate main application file
        :param directory: directory to generate main application file in
        :param class_names_in_exec_order: CPU/GPU partition class names in execution order
        :param gpu_partition_class_names: names of GPU (TensorRT (sub) DNNs)
        :param cpu_partition_class_names: names of CPU (ARM-CL (sub) DNNs)
        :param cpu_core_per_class_name: (dictionary) key (string)=class_name, value(int) = cpu_core_id
         allocation of CPU cores for every partition
         :param inter_partition_connections_desc: json-formatted description of connections between partitions
        :param inter_partition_buffers: inter-partition buffers specification
    """
    filepath = directory + "/" + "appMain.cpp"
    with open(filepath, "w") as print_file:
        visitor = WrapperAppMainGenerator(print_file,
                                          class_names_in_exec_order,
                                          gpu_partition_class_names,
                                          cpu_partition_class_names,
                                          cpu_core_per_class_name,
                                          inter_partition_connections_desc,
                                          inter_partition_buffers)
        visitor.visit()


class WrapperAppMainGenerator(CodegenVisitor):
    """
     * Generator of main application file, which contains
     * application structure and high-level control logic
    """
    def __init__(self, print_file,
                 class_names_in_exec_order: [],
                 gpu_partition_class_names: [],
                 cpu_partition_class_names: [],
                 cpu_core_per_class_name: {},
                 inter_partition_connections_desc: [],
                 inter_partition_buffers: [DataBuffer]):
        """
        Create new application main-file generator
        :param print_file: file to print app-main code in
        :param gpu_partition_class_names: names of GPU (TensorRT (sub) DNNs)
        :param cpu_partition_class_names: names of CPU (ARM-CL (sub) DNNs)
        :param cpu_core_per_class_name: (dictionary) key (string)=class_name, value(int) = cpu_core_id
         allocation of CPU cores for every partition
        :param inter_partition_buffers: (optional) list of buffers, used to store data
            exchanged between DNN partitions
        """
        super().__init__(print_file, prefix="")
        self.class_names_in_exec_order = class_names_in_exec_order
        self.gpu_partition_class_names = gpu_partition_class_names
        self.cpu_partition_class_names = cpu_partition_class_names
        self.cpu_core_per_class_name = cpu_core_per_class_name
        self.default_cpu_core = 1
        self.inter_partition_connections = inter_partition_connections_desc
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
        std_lib_headers = ["thread", "iostream"]
        local_lib_headers = get_static_lib_class_names()
        for class_name in self.class_names_in_exec_order:
            local_lib_headers.append(class_name)

        for lib_header in std_lib_headers:
            self._include_std_cpp_header(lib_header)

        for lib_header in local_lib_headers:
            self._include_local_cpp_header(lib_header)

    def _write_main(self):
        self.write_line("int main (int argc, char **argv) {")
        self.prefix_inc()
        self.write_line("std::cout<<\"***DNN building phase.***\"<<std::endl;")
        self._create_partitions()
        self._create_buffers()
        self._allocate_buffers()
        self._create_pthread_parameters()
        self._inference()
        self._clean_memory()
        self.write_line("")
        self.write_line("return 0;")
        self.prefix_dec()
        self.write_line("}")

    def _create_partitions(self):
        self.write_line("/////////////////////////////////")
        self.write_line("// CREATE PARTITIONS (SUBNETS) //")
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

    def _create_buffers(self):
        self.write_line("//////////////////////////////////////////////////////////////////////////////////////////////")
        self.write_line("// CREATE AND ALLOCATE BUFFERS (Areas of memory storing data exchanged between the subnets) //")
        for buffer in self.inter_partition_buffers:
            buf_class = buf_type_to_buf_class(buffer.type)
            self.write_line(buf_class + " " + buffer.name + ";")
            self.write_line(buffer.name + ".init(" + "\"" + buffer.name + "\"" + ", " + str(buffer.size) + ");")

    def _allocate_buffers(self):
        self.write_line("")
        self.write_line("// Allocate buffers")
        for connection in self.inter_partition_connections:
            buf = self._find_buffer_for_connection(connection)
            if buf is None:
                print("WARNING: no buffer found for connection", connection["name"])
            else:
                src_class_name = connection["src"]
                dst_class_name = connection["dst"]
                src_partition_name = self._get_partition_name(src_class_name)
                dst_partition_name = self._get_partition_name(dst_class_name)
                self.write_line(src_partition_name + ".addOutputBufferPtr(&" + buf.name + ");")
                self.write_line(dst_partition_name + ".addInputBufferPtr(&" + buf.name + ");")
        self.write_line("")

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
        raise Exception("Class name " + class_name + " not defined in the partiiton classes list")

    def _find_buffer_for_connection(self, connection_desc: {}):
        connection_name = connection_desc["name"]
        for buffer in self.inter_partition_buffers:
            if connection_name in buffer.users:
                return buffer
        return None

    def _create_pthread_parameters(self):
        self.write_line("/////////////////////////////////////////////////////////////")
        self.write_line("// PTHREAD thread_infoparams //")
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

        self.write_line("// Allocate memory for pthread_create() arguments")
        self.write_line("const int num_threads = subnets;")
        self.write_line("struct ThreadInfo* thread_info = "
                        "(struct ThreadInfo*)(calloc(num_threads, sizeof(struct ThreadInfo)));")
        self.write_line("")

        self.write_line("// Allocate CPU cores")
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
        self.write_line("std::cout<<\" - Threads creation and execution.\"<<std::endl;")
        self.write_line("")
        self.write_line("//start timer")
        self.write_line("auto start = std::chrono::system_clock::now();")
        self.write_line("std::cout<<\"start timer at \";")
        self.write_line("Timer::printCurrentTimeAndDate();")
        self.write_line("")
        self.write_line("//Create and run posix threads")
        thread_id = 0
        for class_name in self.class_names_in_exec_order:
            partition_name = "p" + str(thread_id)
            if class_name in self.gpu_partition_class_names:
                self.write_line("std::thread my_thread" + str(thread_id) +
                                "(&Subnet::main, &" + partition_name +
                                ", &thread_info[" + str(thread_id) + "]);//(GPU)")
            else:
                self.write_line("std::thread my_thread" + str(thread_id) +
                                "(&Subnet::main, &" + partition_name +
                                ", &thread_info[" + str(thread_id) + "]);//(CPU)")
            thread_id += 1
        self.write_line("")

        self.write_line("//join posix threads")
        for i in range(len(self.class_names_in_exec_order)):
            self.write_line("my_thread" + str(i) + ".join();")
        self.write_line("")
        self.write_line("//stop timer")
        self.write_line("auto end = std::chrono::system_clock::now();")
        self.write_line("std::cout<<\"end timer at \";")
        self.write_line("Timer::printCurrentTimeAndDate();")
        self.write_line("")
        self.write_line("// compute execution time")
        self.write_line("double execTimeS = Timer::timeElapsed(start, end);")
        self.write_line("std::cout<<\"Exec time: \"<<execTimeS<<\" sec\"<<std::endl;")
        self.write_line("")

    def _clean_memory(self):
        """Clean platform memory"""
        self.write_line("/////////////////////////////////////////////////////////////")
        self.write_line("// CLEAN MEMORY //")
        self.write_line("std::cout<<\"*** Memory cleanup ***\"<<std::endl;")
        self.write_line("")
        self.write_line("//delete pthread parameters")
        self.write_line("std::cout<<\" - Pthread parameters destruction\"<<std::endl;")
        self.write_line("free(thread_info);")

