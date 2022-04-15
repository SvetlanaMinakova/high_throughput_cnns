from codegen.codegen_visitor import CodegenVisitor
from models.dnn_model.dnn import DNN
import traceback


def visit_dnn(dnn: DNN, directory, proc_type="CPU"):
    filepath = directory + "/" + dnn.name + ".cpp"
    with open(filepath, "w") as print_file:
        visitor = DNNWrapperCPPVisitor(dnn, print_file, proc_type)
        visitor.visit()


class DNNWrapperCPPVisitor(CodegenVisitor):
    def __init__(self, dnn: DNN, print_file, proc_type:str):
        """
        Create new CPP-code visitor of a DNN/DNN partition
        :param dnn: DNN to visit
        :param print_file: open file to print CPP code of the DNN
        :param profile: include profiling code
        """
        super().__init__(print_file, prefix="")
        self.dnn = dnn
        self.proc_type = proc_type
        self.input_layer = self.dnn.get_input_layer()
        self.output_layer = self.dnn.get_output_layer()
        self.class_name = dnn.name

    def visit(self):
        try:
            self._include_header()
        except Exception:
            print(".cpp file creation error for DNN/partition " + self.class_name)
            traceback.print_exc()

    def _include_header(self):
        self.write_line("#include \"" + self.class_name + ".h\"")

