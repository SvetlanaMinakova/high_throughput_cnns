from codegen.codegen_visitor import CodegenVisitor
from models.dnn_model.dnn import DNN, Layer, layer_has_null_or_empty_pads


def visit_layer(dnn: DNN, layer: Layer, print_file, prefix, end_substream=False):
    layer_input_con = dnn.get_layer_input_connections(layer)
    layer_output_con = dnn.get_layer_output_connections(layer)
    layer_external_inputs = dnn.get_layer_external_inputs(layer)
    visitor = LayerARMCLCPPVisitor(layer, layer_input_con, layer_output_con, layer_external_inputs, print_file, prefix, end_substream)
    visitor.generate_code()


class LayerARMCLCPPVisitor(CodegenVisitor):
    def __init__(self, layer: Layer, input_con, output_con, layer_external_inputs, print_file, prefix,
                 end_sub_stream=False):
        super().__init__(print_file, prefix)
        self.layer = layer
        self.input_con = input_con
        self.output_con = output_con
        self.end_sub_stream = end_sub_stream

        # external I/Os
        self.external_inputs = layer_external_inputs

    def generate_code(self):
        if self.layer.op != "skip":
            layer_inputs_num = len(self.input_con) + len(self.external_inputs)
            if layer_inputs_num > 1:
                self._define_multi_input_layer(layer_inputs_num)
            else:
                self._define_single_input_layer()

    def _define_multi_input_layer(self, inputs_num):
        definition = get_layer_multi_input_definition(self.layer, inputs_num)
        line_id = 0
        for line in definition:
            if line_id == (len(definition)-1) and self.end_sub_stream:
                line += ";"
            self.write_line(line)
            line_id += 1

    def _define_single_input_layer(self):
        definition_lines = get_layer_definition(self.layer)
        self.write_line(definition_lines[0])
        self.prefix_inc()
        self.prefix_inc()
        for definition_line in definition_lines[1:]:
            self.write_line(definition_line)
        self._set_name()
        self.prefix_dec()
        self.prefix_dec()

    def _set_name(self):
        line = ".set_name(\"" + self.layer.name + "\")"
        if self.end_sub_stream:
            line += ";"
        self.write_line(line)


##############################
# SINGLE-INPUT LAYERS #


def get_layer_definition(layer):
    op = layer.op
    if op == "conv":
        definition = get_conv_definition(layer)
        return definition
    if op == "gemm":
        definition = get_gemm_definition(layer)
        return definition
    if op == "pool":
        definition = get_pool_definition(layer)
        return definition
    if op == "activation":
        definition = get_activation_definition(layer)
        return definition

    if op == "normalization":
        definition = get_normalization_definition(layer)
        return definition

    if op == "softmax":
        definition = get_softmax_definition(layer)
        return definition

    if op == "arithmetic":
        definition = get_arithmetic_single_input_definition(layer)
        return definition
    """
    # skip layers are simply skipped
    if op == "skip":
        definition = get_skip_definition(layer)
        return definition
    """
    # default
    print("WARNING: arm-cl codegen: ARM CL definition is unsupported for layer", str(layer))
    definition = get_unknown_layer_definition(layer)
    return definition


def get_conv_definition(layer):
    definition_lines = ["<< ConvolutionLayer(",
                        str(layer.fs) + "U, " + str(layer.fs) + "U, " + str(layer.ofm) + "U,",
                        "get_weights_accessor(data_path, \" \", weights_layout),",
                        "get_weights_accessor(data_path, \" \"),",
                        get_pad_stride_info_line(layer) + ")"
                        ]

    return definition_lines


def get_pool_definition(layer):
    pooling_sub_op = "PoolingType::" + get_pooling_type(layer)
    definition_lines = ["<< PoolingLayer(PoolingLayerInfo(" + pooling_sub_op + ", "
                        + str(layer.fs) + ", " +
                        get_pad_stride_info_line(layer) + ")" + ")"]

    return definition_lines


def get_pooling_type(layer):
    # default
    subop = "MAX"
    if layer.subop:
        if layer.subop in ["reducemean", "averagepool", "globalaveragepool"]:
            subop = "AVG"
    return subop


def get_pad_stride_info_line(layer):
    global_pooling = is_global_pooling(layer)

    # determine stride
    stride = layer.stride if not global_pooling else layer.res

    # determine padding
    w_pad = h_pad = 0
    if not layer_has_null_or_empty_pads(layer):
        w_pad = int((layer.pads[0] + layer.pads[2])/2)
        h_pad = int((layer.pads[1] + layer.pads[3])/2)

    line = "PadStrideInfo(" + str(stride) + ", " + str(stride) + ", " + str(w_pad) + ", " + str(h_pad)

    # set additional parameters for global pooling
    if global_pooling:
        line += ", DimensionRoundingType::CEIL"

    line += ")"

    return line

def is_global_pooling(layer):
    if "global" in layer.subop:
        return True
    if layer.subop == "reducemean":
        return True
    return False


def get_gemm_definition(layer):
    definition_lines = ["<< FullyConnectedLayer(",
                        str(layer.ofm) + "U,"
                        "get_weights_accessor(data_path, \" \", weights_layout),",
                        "get_weights_accessor(data_path, \" \"))"
                        ]
    return definition_lines


def get_activation_definition(layer):
    act_type = get_activation_type(layer)
    definition_lines = ["<< ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::" + act_type + "))"]
    return definition_lines


def get_activation_type(layer):
    # default activation type
    # used for CLIP operators as well
    activation_type = "RELU"
    if layer.subop == "leaky_relu":
        return "LEAKY_RELU"
    if layer.subop == "sigmoid":
        return "LOGISTIC"

    return activation_type


def get_softmax_definition(layer):
    definition_lines = ["<< SoftmaxLayer()"]
    return definition_lines


def get_normalization_definition(layer):
    """visit normalization layer"""
    if layer.subop in ["bn", "batchnormalization"]:
        definition = get_bn_definition(layer)
        return definition
    if layer.subop in ["lrn"]:
        definition = get_lrn_definition(layer)
        return definition
    definition = get_unknown_layer_definition(layer)
    return definition


def get_bn_definition(layer):
    definition_lines = ["<< BatchNormalizationLayer(get_weights_accessor(data_path, \" \"),",
                        "get_weights_accessor(data_path, \"\"),",
                        "get_weights_accessor(data_path, \"\"),",
                        "get_weights_accessor(data_path, \"\"),",
                        "0.0010000000474974513f)"
                        ]
    return definition_lines


def get_lrn_definition(layer):
    # TODO: extract from ONNX node
    size = 5
    alpha = 0.0001
    beta = 0.75
    bias = 1
    definition_lines = ["<< NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, " +
                        str(size) + ", " + str(alpha) + "f, " + str(beta) + "f))"]
    return definition_lines


def get_arithmetic_single_input_definition(layer):
    # TODO: unsupported! replace by real definition!!!
    definition_lines = ["<< BatchNormalizationLayer(get_weights_accessor(data_path, \" \"),",
                        "get_weights_accessor(data_path, \"\"),",
                        "get_weights_accessor(data_path, \"\"),",
                        "get_weights_accessor(data_path, \"\"),",
                        "0.0010000000474974513f)"
                        ]
    return definition_lines


######################
# MULTI-INPUT LAYERS #

def get_layer_multi_input_definition(layer, layer_inputs_num):
    if layer_inputs_num < 2:
        raise Exception("ARM-CL codegen ERROR: multi-input_examples layer" + layer.name + " has "
                        + str(layer_inputs_num) + "inputs, while at least 2 inputs are expected")
    op = layer.op
    if op == "arithmetic":
        definition = get_arithmetic_multi_input_definition(layer, layer_inputs_num)
        return definition
    if op == "concat":
        definition = get_concat_definition(layer, layer_inputs_num)
        return definition

    # default
    print("WARNING: tensorrt codegen: TensorRT MULTI-INPUT definition is unsupported for layer", str(layer))
    definition = get_unknown_layer_definition(layer)
    return definition


def get_arithmetic_multi_input_definition(layer, layer_inputs_num):
    """visit arithmetic layer which has multiple (two) inputs"""
    op = get_element_wise_op(layer)
    definition_line = "graph << EltwiseLayer(std::move(sub_stream0)," +\
                      " std::move(sub_stream1), EltwiseOperation::" + op +\
                      ").set_name(\"" + layer.name + "\")"
    definition_lines = [definition_line]
    return definition_lines


def get_element_wise_op(layer):
    """
     Translate Arithmetic (element wise) operator name from ONNX namespace to TENSORRT namespace
     :param layer Arithmetic (element wise) layer
     :return Arithmetic operator in TENSORRT namespace
    """
    # default
    element_wise = "Add"
    if layer.subop == "add":
        return "Add"

    return element_wise


def get_concat_definition(layer, layer_inputs_num):
    """visit concat layer"""
    """visit arithmetic layer which has multiple (two) inputs"""
    op = get_element_wise_op(layer)
    definition_line = "graph << ConcatLayer("
    for layer_input_id in range(layer_inputs_num - 1):
        definition_line += "std::move(sub_stream" + str(layer_input_id) + "), "
    definition_line += "std::move(sub_stream" + str(layer_inputs_num-1) + ")"
    definition_line += ").set_name(\"" + layer.name + "\")"
    definition_lines = [definition_line]
    return definition_lines


##############################
# UNKNOWN/UNSUPPORTED LAYERS #

def get_unknown_layer_definition(layer):
    definition_lines = ["//ERROR LAYER " + layer.name + " OF UNKNOWN TYPE " + layer.op]
    return definition_lines


def get_moc_layer_definition(layer):
    relu_layer = Layer(res=layer.res, op="relu", fs=layer.fs, ifm=layer.ifm, ofm=layer.ofm, bordermode="same")
    definition = get_activation_definition(relu_layer)
    return definition

