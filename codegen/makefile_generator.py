from codegen.codegen_visitor import CodegenVisitor
from codegen.codegen_config import get_config
"""
//////////////////////////           MAKEFILE                 ////////////////////////
Make-file generator for tensorrt/arm-cl code
"""


def generate_makefile(directory,
                      gpu_partition_class_names: [],
                      cpu_partition_class_names: [],
                      arm_cl: bool,
                      trt: bool):
    """
        generate tensorrt/arm-cl application makefile
        :param directory: directory to generate makefile in
        :param gpu_partition_class_names: names of GPU (TensorRT (sub) DNNs)
        :param cpu_partition_class_names: names of CPU (ARM-CL (sub) DNNs)
        :param arm_cl: (flag), if True: use ARM-CL (CPU) code and libraries
        :param trt: (flag), if True: use TensorRT (GPU) code and libraries
    """
    filepath = directory + "/" + "Makefile"
    with open(filepath, "w") as print_file:
        visitor = MakefileGenerator(print_file, gpu_partition_class_names, cpu_partition_class_names, arm_cl, trt)
        visitor.visit()


class MakefileGenerator(CodegenVisitor):

    def __init__(self, print_file, gpu_partition_class_names: [], cpu_partition_class_names: [], arm_cl: bool, trt: bool):
        """
        tensorrt/arm-cl application makefile creator
        :param print_file: file to generate makefile in
        :param gpu_partition_class_names: names of GPU (TensorRT (sub) DNNs)
        :param cpu_partition_class_names: names of CPU (ARM-CL (sub) DNNs)
        :param arm_cl: (flag), if True: use ARM-CL (CPU) code and libraries
        :param trt: (flag), if True: use TensorRT (GPU) code and libraries
        """
        super().__init__(print_file, prefix="")
        self.gpu_partition_class_names = gpu_partition_class_names
        self.cpu_partition_class_names = cpu_partition_class_names
        self.custom_buffers_class_names = ["Subnet", "SingleBuffer", "DoubleBuffer", "SharedBuffer", "types"]
        self.arm_cl = arm_cl
        self.trt = trt
        self.config = get_config()
        self.arm_cl_path = self.config["arm_cl_path"]
        self.cuda_path = self.config["cuda_path"]
        self.cpp_standard = self.config["cpp_standard"]

    def visit(self):  
        try:
            self.write_lib_paths_and_flags()
            self.write_objects()
            self.write_line("")
            self.write_line("")
            self.write_line("PRG = appMain")
            self.write_line("")
            self.write_line("all: ${PRG}")
            self.write_line("")

            # build main classes compiler string
            self.write_line("appMain: ${OBJS}")
            self.write("\t${CXX} -o appMain ${OBJS}")

            app_main_compiler_dependencies = self.build_app_main_compiler_dependencies()
            self.write_line(app_main_compiler_dependencies)
            self.write_line("")

            # classes generation
            self.write_per_object_classes()

            # custom code
            for class_name in self.custom_buffers_class_names:
                self.write_line(class_name + ".o: " + class_name + ".cpp")
                self.write_line("\t${CXX} ${CXXFLAGS} -c -g $?")
                self.write_line("")

            """
            self.write_line("fifo.o: fifo.cpp")
            self.write_line("\t${CXX} ${CXXFLAGS} -c -g $?")
            self.write_line("")
            """

            self.write_line("appMain.o: appMain.cpp")
            self.write("\t${CXX} appMain.cpp")
            self.write(app_main_compiler_dependencies)
            self.write_line(" -c -g $?")
            self.write_line("")

            self.write_line("")
            self.write_line("")
            self.write_line("clean:")
            self.write_line("\trm -rf *~ *.o ")

        except Exception:
            print("ERROR: Cannot create the default ARM-CL/TensorRT makefile. Please supply your own makefile")

    def write_lib_paths_and_flags(self):
        self.write_line("CXX=aarch64-linux-gnu-g++")
        self.write_line("")
        self.write_line("CXXFLAGS= -std=c++" + str(self.cpp_standard) + " -Wl,--allow-shlib-undefined")
        self.write_line("")

        if self.arm_cl:
            self.write_line("ARM_PATH=" + self.arm_cl_path)
            self.write_line("")
            self.write_line("ARM_INCFLAG = -I${ARM_PATH} -I${ARM_PATH}/include")
            self.write_line("")
            self.write_line("ARM_SOURCES = ${ARM_PATH}/utils/Utils.cpp "
                            "${ARM_PATH}/utils/GraphUtils.cpp ${ARM_PATH}/utils/CommonGraphOptions.cpp")
            self.write_line("CXXLIB = -L${ARM_PATH} -larm_compute_graph -larm_compute -larm_compute_core -lpthread")
            self.write_line("")

        else:
            self.write_line("CXXLIB = -lpthread")
            self.write_line("")

        if self.trt:
            self.write_line("CUDA_FLAGS= -Wall -std=c++" + str(self.cpp_standard) + " -O2")
            self.write_line("")
            self.write_line("CUDA_PATH=" + self.cuda_path + "")
            self.write_line("")
            self.write_line(
                "CUDA_INCFLAG = -I\"" + self.cuda_path +
                "/include\" -I\"/usr/local/include\" -I\"../include\" -I\"../common\"" +
                " -I\"" + self.cuda_path + "/include\" -I\"../../include\"  -D_REENTRANT")
            self.write_line("")
            self.write_line(
                "CUDA_LIB = -L\"\" -L\"" + self.cuda_path +
                "/targets/x86_64-linux/lib64\" -L\"/usr/local/lib\" -L\"../lib\"" +
                " -L\"" + self.cuda_path + "/lib64\" -L\"" + self.cuda_path + "/lib64\" -L\"../../lib\"" +
                "   -L./ -Wl,--start-group -lnvinfer -lnvparsers -lnvinfer_plugin -lcudnn -lcublas "
                "-lcudart_static -lnvToolsExt -lcudart -lrt -ldl -lpthread -Wl,--end-group")
            self.write_line("")

    def write_objects(self):
        self.write_line("")
        self.write("OBJS = ")

        # CPU part
        if self.arm_cl:
            for class_name in self.cpu_partition_class_names:
                self.write(class_name + ".o ")

            self.write("cpu_engine.o ")

            arm_src_classes = _get_arm_cl_src_classes()
            for class_name in arm_src_classes:
                self.write(class_name + ".o ")

        # GPU part
        if self.trt:
            self.write("gpu_engine.o ")
            for class_name in self.gpu_partition_class_names:
                self.write(class_name + ".o ")
            self.write("gpu_partition.o ")

        # custom buffers
        for class_name in self.custom_buffers_class_names:
            self.write(class_name + ".o ")

        self.write("appMain.o")#"fifo.o appMain.o"

    def build_app_main_compiler_dependencies(self):
        app_main_compiler_dependencies = ""

        if self.arm_cl:
            app_main_compiler_dependencies += " $(ARM_INCFLAG)"

        if self.trt:
            app_main_compiler_dependencies += " $(CUDA_INCFLAG)"
        app_main_compiler_dependencies += " ${CXXLIB}"

        if self.trt:
            app_main_compiler_dependencies += " ${CUDA_LIB}"
        app_main_compiler_dependencies += " ${CXXFLAGS}"
        return app_main_compiler_dependencies

    def write_per_object_classes(self):
        # per - object classes
        # CPU
        if self.arm_cl:
            arm_src_classes = _get_arm_cl_src_classes()
            for class_name in arm_src_classes:
                self.write_line(class_name + ".o: ")
                self.write_line(
                    "\t${CXX} ${ARM_PATH}/utils/" + class_name + ".cpp $(ARM_INCFLAG) ${CXXLIB} ${CXXFLAGS} -c -g $?")
                self.write_line("")

            self.write_line("cpu_engine.o: ")
            self.write_line("\t${CXX} cpu_engine.cpp $(ARM_INCFLAG) ${CXXLIB} ${CXXFLAGS} -c -g $?")
            self.write_line("")

            for class_name in self.cpu_partition_class_names:
                self.write_line(class_name + ".o: ")
                self.write_line("\t${CXX} " + class_name + ".cpp $(ARM_INCFLAG) ${CXXLIB} ${CXXFLAGS} -c -g $?")
                self.write_line("")

        # GPU
        if self.trt:
            self.write_line("gpu_partition.o: ")
            self.write_line("\t${CXX} gpu_partition.cpp $(CUDA_INCFLAG) ${CUDA_LIB} ${CUDA_FLAGS} -c -g $?")
            self.write_line("")

            self.write_line("gpu_engine.o: ")
            self.write_line("\t${CXX} gpu_engine.cpp $(CUDA_INCFLAG) ${CUDA_LIB} ${CUDA_FLAGS} -c -g $?")
            self.write_line("")

            for class_name in self.gpu_partition_class_names:
                self.write_line(class_name + ".o: ")
                self.write_line("\t${CXX} " + class_name + ".cpp $(CUDA_INCFLAG) ${CUDA_LIB} ${CUDA_FLAGS} -c -g $?")
                self.write_line("")


def _get_arm_cl_src_classes():
    """ Get list of ARM-CL Lib source classes used for application building"""
    arm_cl_src_classes = ["Utils", "GraphUtils", "CommonGraphOptions"]
    return arm_cl_src_classes


