from codegen.codegen_visitor import CodegenVisitor
from models.dnn_model.dnn import DNN
import traceback


def visit_dnn(dnn: DNN, directory, proc_type="CPU"):
    filepath = directory + "/" + dnn.name + ".h"
    with open(filepath, "w") as print_file:
        visitor = DNNWrapperHVisitor(dnn, print_file, proc_type)
        visitor.visit()


class DNNWrapperHVisitor(CodegenVisitor):
    def __init__(self, dnn: DNN, print_file, proc_type):
        """
        Create new H-code visitor of a DNN/DNN partition
        :param dnn: DNN to visit
        :param print_file: open file to print CPP code of the DNN
        :param proc_type: type of target processor
        """
        super().__init__(print_file, prefix="")
        self.dnn = dnn
        self.proc_type = proc_type
        self.input_layer = self.dnn.get_input_layer()
        self.output_layer = self.dnn.get_output_layer()
        self.class_name = dnn.name
        self.base_class_name = "Subnet"

    def visit(self):
        self._write_common_beginning()
        self._write_common_end()

    def _write_common_beginning(self):
        """
        Begin a header file with common beginning
        """
        name = self.class_name
        self.write_line("// File automatically generated by ESPAM")
        self.write_line("")
        self.write_line("#ifndef " + name + "_H")
        self.write_line("#define " + name + "_H")
        self.write_line("")
        self.write_line("#include \"" + self.base_class_name + ".h\"")
        self.write_line("")
        self.write_line("class " + name + " : public " + self.base_class_name + " {")
        self.write_line("public:")
        self.prefix_inc()
        self._write_constructor()
        self.prefix_dec()

    def _write_constructor(self):
        self.write_line("explicit " + self.class_name + "(int runs=1, int execDelayMS=0, int rwDelayMS=0): " +
                        self.base_class_name + "(\"" + self.class_name + "\", runs, execDelayMS, rwDelayMS){}")

    def _write_common_end(self):
        """
        Finish a header file with common ending
        """
        self.write_line("};")
        self.write_line("#endif // " + self.base_class_name + "_H")

