from codegen.codegen_visitor import CodegenVisitor
from models.dnn_model.dnn import Layer


def visit_dnn_io(external_io, io_id, print_file, prefix, is_input):
    """
    Visit DNN external data source/consumer
    :param external_io: explicitly specified data source/consumer
    :param print_file: file to print code in
    :param prefix: code prefix (tabulation)
    :param is_input: (flag) is this input_examples or output?
    :param io_id: index of input_examples (for multi-input_examples DNNs/DNN partititions)
    """
    visitor = CPPExternalIOSVisitor(external_io, io_id, print_file, prefix, is_input)
    visitor.visit()


def simulate_dnn_io(layer: Layer, print_file, prefix, is_input):
    """
    Simulate (fake) external data source/consumer for a CNN layer
    :param layer: CNN layer
    :param print_file: file to print code in
    :param prefix: code prefix (tabulation)
    :param is_input: (flag) is this input_examples or output?
    """
    visitor = CPPSimulatedExternalIOSVisitor(layer, print_file, prefix, is_input)
    visitor.visit()


class CPPExternalIOSVisitor(CodegenVisitor):
    """
    Visitor of DNN explicitly specified I/O s
    """
    def __init__(self, external_io, io_id, print_file, prefix, is_input=True):
        super().__init__(print_file, prefix)
        self.external_io = external_io
        self.io_id = io_id
        self.data_layer = external_io.data_layer
        self.dnn_layer = external_io.dnn_layer
        self.is_input = is_input

    def visit(self):
        if self.is_input:
            self.write_input()
        else:
            self.write_output()

    def write_input(self):
        # process this layer as the input_examples layer of a DNN/partition
        # TODO: incorporate I/O id for additional inputs
        self.write_line("// input_examples data " + self.data_layer.name)
        input_blob_name = "this->INPUT_BLOB_NAME"
        if self.io_id > 1:
            input_blob_name = self.data_layer.name
        self.write_line("ITensor* " + self.data_layer.name +
                        " = network->addInput(" + input_blob_name + ","
                        " dt, Dims4{1, " + str(self.data_layer.ofm)
                        + ", " + str(self.data_layer.oh)
                        + ", " + str(self.data_layer.ow) + "});")

        self.write_line("assert(" + self.data_layer.name + ");")
        self.write_line("")
        # _simulatedDataLayers.add(inputLayerNameDef);

    def write_output(self):
        # process this layer as the output layer of a DNN/partition
        self.write_line("")
        self.write_line("// process layer " + self.dnn_layer.name + " as output ")
        self.write_line(self.dnn_layer.name + "->getOutput(0)->setName(this->OUTPUT_BLOB_NAME);")
        self.write_line("network->markOutput(*" + self.dnn_layer.name + "->getOutput(0));")


class CPPSimulatedExternalIOSVisitor(CodegenVisitor):
    """
    Visitor of DNN I/O layers for those DNNs that have no explicit external I/Os
    """
    def __init__(self, layer: Layer, print_file, prefix, is_input=True):
        super().__init__(print_file, prefix)
        self.layer = layer
        self.is_input = is_input

    def visit(self):
        if self.is_input:
            self.simulate_input_data_layer()
        else:
            self.simulate_output_data_layer()

    def simulate_input_data_layer(self):
        input_name = self.layer.name
        # give the input_examples a unique name
        if self.layer.op != "data":
            input_name += "_input"
        # process this layer as the input_examples layer of a DNN/partition
        self.write_line("// moc input_examples data " + input_name)
        self.write_line("ITensor* " + input_name +
                        " = network->addInput(this->INPUT_BLOB_NAME,"
                        " dt, Dims4{1, this->INPUT_C, this->INPUT_H, this->INPUT_W});")
        # self.write_line("assert(" + self.layer.name + ");")
        # _simulatedDataLayers.add(inputLayerNameDef);

    def simulate_output_data_layer(self):
        # process this layer as the output layer of a DNN/partition
        self.write_line("//moc output data " + self.layer.name)
        self.write_line(self.layer.name + "->getOutput(0)->setName(this->OUTPUT_BLOB_NAME);")
        self.write_line("network->markOutput(*" + self.layer.name + "->getOutput(0));")

