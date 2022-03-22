import os.path
from fileworkers.common_fw import create_or_overwrite_dir
from util import get_project_root
from shutil import copyfile


class CodegenVisitor:
    """
    Visitor, used for code generation
    """
    def __init__(self, print_file, prefix=""):
        self.print_file = print_file
        self.prefix = prefix
        self.__prefix_len = 2

    def write_line(self, line):
        self.print_file.write(self.prefix + line + "\n")

    def write(self, line):
        self.print_file.write(line)

    def prefix_inc(self):
        for _ in range(self.__prefix_len):
            self.prefix += " "

    def prefix_dec(self):
        for _ in range(self.__prefix_len):
            if len(self.prefix) > 0:
                self.prefix = self.prefix[-1]

    def _include_std_cpp_header(self, header):
        self.write_line("#include <" + header + ">")

    def _include_local_cpp_header(self, header):
        self.write_line("#include \"" + header + ".h\"")

    def _include_namespace(self, namespace):
        self.write_line("using namespace " + namespace + ";")


def copy_static_app_code(code_dir):
    """
    Copy static (hand-written, DNN-independent) code for ARM-CL/TRT code execution
    :param code_dir: target code directory
    """
    static_files_dir = os.path.join(str(get_project_root()), "codegen/static_lib_files/app")
    if os.path.exists(static_files_dir):
        for the_file in os.listdir(static_files_dir):
            src = os.path.join(static_files_dir, the_file)
            dst = os.path.join(code_dir, the_file)
            copyfile(src, dst)
        print("Static files successfully copied")
        print(" - from " + static_files_dir)
        print(" - to " + code_dir)
    else:
        print("WARNING: codegen: no static lib files copied. ")
        print("reason: I could not find static files directory: " + static_files_dir)
