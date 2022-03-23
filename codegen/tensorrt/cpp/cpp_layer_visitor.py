from codegen.codegen_visitor import CodegenVisitor
from models.dnn_model.dnn import DNN, Layer, layer_has_null_or_empty_pads


def visit_layer(dnn: DNN, layer: Layer, print_file, prefix):
    layer_input_con = dnn.get_layer_input_connections(layer)
    layer_output_con = dnn.get_layer_output_connections(layer)
    layer_external_inputs = dnn.get_layer_external_inputs(layer)
    visitor = LayerTRTCPPVisitor(layer, layer_input_con, layer_output_con, layer_external_inputs, print_file, prefix)
    visitor.generate_code()


class LayerTRTCPPVisitor(CodegenVisitor):
    def __init__(self, layer: Layer, input_con, output_con, layer_external_inputs, print_file, prefix):
        super().__init__(print_file, prefix)
        self.layer = layer
        self.input_con = input_con
        self.output_con = output_con

        # external I/Os
        self.external_inputs = layer_external_inputs

    def generate_code(self):
        self.write_line("//" + self.layer.name)
        layer_inputs_num = len(self.input_con) + len(self.external_inputs)
        if layer_inputs_num > 1:
            layer_inputs = get_layer_inputs(self.layer, self.input_con, self.external_inputs)
            definition = get_layer_multi_input_definition(self.layer, layer_inputs)
            self.write_line(definition)
        else:
            layer_input = get_layer_input(self.layer, self.input_con, self.external_inputs)
            definition = get_layer_definition(self.layer, layer_input)
            self.write_line(definition)

        if self.layer.op != "skip":
            self._write_assert()
            self._set_name()
            self._set_stride()
            self._process_pads()
        # print("Layer", self.layer.name, "processed")

    # check if layer is properly defined
    def _write_assert(self):
        self.write_line("assert(" + self.layer.name + ");")

    # assign layer a name
    def _set_name(self):
        self.write_line(self.layer.name + "->setName(\"" + self.layer.name + "\");")

    def _set_stride(self):
        if self.layer.op in ["conv", "pool"]:
            stride = self.layer.stride if self.layer.stride else 1
            if self.layer.subop:
                if "global" in self.layer.subop:
                    stride = self.layer.res
            self.write_line(self.layer.name + "->setStride(DimsHW{" + str(stride) + ", "
                            + str(stride) + "});")

    def _process_pads(self):
        """
        Process pads of the layer
         * Pads are values, added to the beginning and ending along each axis
         * to avoid "inconvenient" data formats in Convolutional and Pooling layers
         * pads have format [x1_begin, x2_begin...x1_end, x2_end,...],
         * where xi_begin the number of pixels added at the beginning of axis `i` and xi_end,
         * the number of pixels added at the end of axis `i`.
         * Pads should contain values >=0
        """
        if not layer_has_null_or_empty_pads(self.layer):
            pads = self.layer.pads
            pads_x = int((pads[1] + pads[3])/2)
            pads_y = int((pads[0] + pads[2])/2)
            self.write_line(self.layer.name + "->setPadding(DimsHW{" + str(pads_x) + ", " + str(pads_y) + "});")
        else:
            # simulate "same" border mode with pads
            if self.layer.get_border_mode() == "same" and self.layer.op in ["conv"]:
                pads_x = int((self.layer.iw * (self.layer.stride - 1) - self.layer.stride + self.layer.fs)/2)
                pads_y = int((self.layer.ih * (self.layer.stride - 1) - self.layer.stride + self.layer.fs)/2)
                self.write_line(self.layer.name + "->setPadding(DimsHW{" + str(pads_x) + ", " + str(pads_y) + "});")


"""
Common functions
"""


def get_layer_input(layer: Layer, input_connections, external_inputs):
    """
         Get string that defines single layer input_examples
         :param layer layer
         :param input_connections: input_examples connections of the layer
         :param external_inputs: external data sources, from outside of the DNN
         :return name of the input_examples buffer
    """
    layer_input = "nullptr"
    if len(input_connections) > 0:
        first_inp_connection = input_connections[0]
        src = first_inp_connection.src
        layer_input = src.name + "->getOutput(0)"
    else:
        if len(external_inputs) > 0:
            first_inp = external_inputs[0]
            src = first_inp.data_layer
            layer_input = src.name
        else:
            layer_input = layer.name + "_input"

    return layer_input


def get_layer_inputs(layer: Layer, input_connections, external_inputs):
    """
         Get a set of strings that defines layer inputs
         :param layer layer
         :param input_connections: input_examples connections of the layer
         :param external_inputs: external data sources, from outside of the DNN
         :return name of the input_examples buffer
    """
    layer_inputs = []
    # layer input_examples = connection within dnn
    for connection in input_connections:
        src = connection.src
        layer_input = src.name
        layer_input = layer_input + "->getOutput(0)"
        layer_inputs.append(layer_input)

    for external_input in external_inputs:
        src = external_input.data_layer
        layer_input = src.name
        layer_inputs.append(layer_input)

    return layer_inputs

#############################################
#   Operator-dependent layer definitions    #

##############################
# SINGLE-INPUT LAYERS #


def get_layer_definition(layer, layer_input):
    op = layer.op
    if op == "conv":
        definition = get_conv_definition(layer, layer_input)
        return definition
    if op == "gemm":
        definition = get_gemm_definition(layer, layer_input)
        return definition
    if op == "pool":
        definition = get_pool_definition(layer, layer_input)
        return definition
    if op == "activation":
        definition = get_activation_definition(layer, layer_input)
        return definition
    if op == "normalization":
        definition = get_normalization_definition(layer, layer_input)
        return definition
    if op == "softmax":
        definition = get_softmax_definition(layer, layer_input)
        return definition
    if op == "arithmetic":
        definition = get_arithmetic_single_input_definition(layer, layer_input)
        return definition
    if op == "skip":
        definition = get_skip_definition(layer, layer_input)
        return definition
    # default
    print("WARNING: tensorrt codegen: TensorRT definition is unsupported for layer", str(layer))
    definition = get_unknown_layer_definition(layer)
    return definition


def get_conv_definition(layer, layer_input):
    # TODO: check TF codegen for depthwise-separable convolutions
    definition = "IConvolutionLayer* " + layer.name + " = network->addConvolution(*" + layer_input + \
                 ", " + str(layer.ofm) + ", DimsHW{" + str(layer.fs) + ", " + str(layer.fs) + "}," +\
                 " weightMap[\"" + layer.name + "_weights\"], weightMap[\"" + layer.name + "_bias\"]);"
    return definition


def get_activation_definition(layer, layer_input):
    """visit activation (nonlinear) layer"""
    activation_type = get_activation_type(layer)
    definition = "IActivationLayer* " + layer.name + "= network->addActivation(*" + layer_input + \
                 ", ActivationType::" + activation_type + ");"
    return definition


def get_activation_type(layer):
    # default activation type
    # used for LeakyRelU and CLIP operators as well
    activation_type = "kRELU"
    if layer.subop == "sigmoid":
        return "kSIGMOID"

    return activation_type


def get_arithmetic_single_input_definition(layer, layer_input):
    """visit arithmetic layer which has one input_examples"""
    definition = "auto " + layer.name + " = network->addScale(*" + layer_input + ", ScaleMode::kUNIFORM," +\
                 " weightMap[\"" + layer.name + "_shift\"], weightMap[\"" + layer.name + "_scale\"], weightMap[\"" + layer.name + "_power\"]);"
    return definition


def get_normalization_definition(layer, layer_input):
    """visit normalization layer"""
    if layer.subop in ["bn", "batchnormalization"]:
        definition = get_bn_definition(layer, layer_input)
        return definition
    if layer.subop in ["lrn"]:
        definition = get_lrn_definition(layer, layer_input)
        return definition
    definition = get_unknown_layer_definition(layer)
    return definition


def get_bn_definition(layer, layer_input):
    """visit batch normalization layer"""
    definition = "auto " + layer.name + " = network->addScale(*" + layer_input + ", ScaleMode::kUNIFORM," +\
                 " weightMap[\"" + layer.name + "_shift\"], weightMap[\"" + layer.name + "_scale\"], weightMap[\"" + layer.name + "_power\"]);"
    return definition


def get_lrn_definition(layer, layer_input):
    # TODO: extract from ONNX node
    size = 5
    alpha = 0.0001
    beta = 0.75
    bias = 1
    definition = "ILRNLayer* " + layer.name + "= network->addLRN(*" + layer_input + "," +\
                 str(size) + "," + str(alpha) + "," + str(beta) + ", " + str(bias) + ");"
    return definition


def get_softmax_definition(layer, layer_input):
    definition = "ISoftMaxLayer* " + layer.name + "= network->addSoftMax(*" + layer_input + ");"
    return definition


def get_gemm_definition(layer, layer_input):
    definition = "IFullyConnectedLayer* " + layer.name + \
                 " = network->addFullyConnected(*" + layer_input + ", " + str(layer.ofm) +\
                 ", weightMap[\"" + layer.name + "_weights\"], weightMap[\"" + layer.name + "_bias\"]);"
    return definition


def get_pool_definition(layer, layer_input):
    func = get_trt_pooling_subop(layer)
    definition = "IPoolingLayer* " + layer.name + " = network->addPooling(*" + layer_input +\
                 ", PoolingType::" + func + ", DimsHW{" + str(layer.fs) + ", " + str(layer.fs) + "});"
    return definition


def get_trt_pooling_subop(layer):
    # default
    subop = "kMAX"
    if layer.subop:
        if layer.subop in ["reducemean", "averagepool", "globalaveragepool"]:
            subop = "kAVERAGE"
    return subop


def get_skip_definition(layer, layer_input):
    """
    Visit skip-layer. Skip-s are layers that do nothing
    during inference (e.g. dropout layers). In this codegen,
    such layers simply point out to previous layer
    """
    prev_layer = layer_input
    if "->getOutput(0)" in layer_input:
        prev_layer = prev_layer.replace("->getOutput(0)", "")
    definition = "auto " + layer.name + " = " + prev_layer + ";"
    return definition


######################
# MULTI-INPUT LAYERS #

def get_layer_multi_input_definition(layer, layer_inputs):
    if len(layer_inputs) < 2:
        raise Exception("TensorRT codegen ERROR: multi-input_examples layer" + layer.name + " has "
                        + str(len(layer_inputs)) + "inputs, while at least 2 inputs are expected")
    op = layer.op
    if op == "arithmetic":
        definition = get_arithmetic_multi_input_definition(layer, layer_inputs)
        return definition
    if op == "concat":
        definition = get_concat_definition(layer, layer_inputs)
        return definition
    
    # default
    print("WARNING: tensorrt codegen: TensorRT MULTI-INPUT definition is unsupported for layer", str(layer))
    definition = get_unknown_layer_definition(layer)
    return definition


def get_arithmetic_multi_input_definition(layer, layer_inputs):
    """visit arithmetic layer which has multiple (two) inputs"""
    element_wise_op = get_element_wise_op(layer)
    definition = "auto " + layer.name + " = network->addElementWise(*" + layer_inputs[0] +\
                 ", *" + layer_inputs[1] + ", ElementWiseOperation::" + element_wise_op + " );"
    return definition


def get_element_wise_op(layer):
    """
     Translate Arithmetic (element wise) operator name from ONNX namespace to TENSORRT namespace
     :param layer Arithmetic (element wise) layer
     :return Arithmetic operator in TENSORRT namespace
    """
    # default
    element_wise = "kSUM"
    if layer.subop == "add":
        return "kSUM"

    if layer.subop in ["mul", "multiply"]:
        return "kPROD"

    if layer.subop in ["sub"]:
        return "kSUB"

    return element_wise


def get_concat_definition(layer, layer_inputs):
    """visit concat layer"""
    inputs_num_str = str(len(layer_inputs))
    definition = "std::array<nvinfer1::ITensor*, " + inputs_num_str + "> " + layer.name + "_inputs{{"
    for layer_input_id in range(len(layer_inputs)-1):
        layer_input = layer_inputs[layer_input_id]
        definition += layer_input + ", "
    definition += layer_inputs[-1] + "}};\n"
    definition += "IConcatenationLayer* " + layer.name + " = network->addConcatenation(" +\
                  layer.name + "_inputs.data(), " + inputs_num_str + ");"
    return definition


##############################
# UNKNOWN/UNSUPPORTED LAYERS #


def get_unknown_layer_definition(layer):
    definition = "//ERROR: LAYER " + layer.name + " OF UNKNOWN TYPE " + layer.op
    return definition

