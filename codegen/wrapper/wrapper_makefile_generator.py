from codegen.codegen_visitor import CodegenVisitor
from codegen.wrapper.static_lib_info import get_static_lib_class_names

"""
//////////////////////////           MAKEFILE                 ////////////////////////
Make-file generator for tensorrt/arm-cl code
"""


def generate_makefile(directory, class_names_in_exec_order):
    """
        generate tensorrt/arm-cl application makefile
        :param directory: directory to generate makefile in
        :param class_names_in_exec_order: classes (subnets) names in execution order
    """
    filepath = directory + "/" + "Makefile"
    with open(filepath, "w") as print_file:
        visitor = WrapperMakefileGenerator(print_file, class_names_in_exec_order)
        visitor.visit()


class WrapperMakefileGenerator(CodegenVisitor):

    def __init__(self, print_file, class_names_in_exec_order):
        """
        tensorrt/arm-cl application makefile creator
        :param print_file: file to generate makefile in
        :param class_names_in_exec_order: classes (subnets) names in execution order
        """
        super().__init__(print_file, prefix="")
        self.class_names = class_names_in_exec_order

    def visit(self):
        try:
            self.write_flags()
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

    def write_flags(self):
        # self.write_line("CXX=aarch64-linux-gnu-g++")
        # self.write_line("")
        self.write_line("CXXFLAGS= -std=c++14")
        self.write_line("")
        self.write_line("CXXLIB = -pthread") # self.write_line("CXXLIB = -lpthread")
        self.write_line("")

    def write_objects(self):
        self.write_line("")
        self.write("OBJS = ")

        local_lib_classes = get_static_lib_class_names()
        for class_name in local_lib_classes:
            self.write(class_name + ".o ")

        # subnet classes
        for class_name in self.class_names:
            self.write(class_name + ".o ")

        self.write("appMain.o")

    def build_app_main_compiler_dependencies(self):
        app_main_compiler_dependencies = ""
        app_main_compiler_dependencies += " ${CXXLIB}"
        app_main_compiler_dependencies += " ${CXXFLAGS}"
        return app_main_compiler_dependencies

    def write_per_object_classes(self):
        for class_name in self.class_names:
            self.write_line(class_name + ".o: ")
            self.write_line("\t${CXX} " + class_name + ".cpp ${CXXLIB} ${CXXFLAGS} -c -g $?")
            self.write_line("")

        # per - object classes
        static_lib_classes = get_static_lib_class_names()
        for class_name in static_lib_classes:
            self.write_line(class_name + ".o: ")
            self.write_line("\t${CXX} " + class_name + ".cpp ${CXXLIB} ${CXXFLAGS} -c -g $?")
            self.write_line("")

