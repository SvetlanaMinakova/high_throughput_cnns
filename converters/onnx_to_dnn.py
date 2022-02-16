import onnx
from onnx import shape_inference
from models.dnn_model.dnn import DNN, Layer, layer_has_null_or_empty_pads
import traceback
import sys
from fileworkers.onnx_fw import read_onnx

"""
This module parses a CNN model in ONNX format and represents it as a DNN class, defined in dnn_model.dnn_model.py
"""


def read_and_check_consistency(path, verbose=True):
    """
    Read an onnx model and ckeck it's consistency. If model is inconsistent,
    the errors stack will be printed. Otherwise "check passed" will be printed.
    :param path: path to onnx model
    :param verbose: print details
    """
    try:
        onnx_model = read_onnx(path)
        onnx_model = set_dataflow(onnx_model, True)
        if verbose:
            print("ONNX MODEL IS CONSISTENT")
        return onnx_model
    except Exception as err:
        if verbose:
            print("ONNX MODEL IS INCONSISTENT")
        traceback.print_tb(err.__traceback__)
        return None


def onnx_to_dnn(onnx_model, check_dnn_consistency=True, data_layout="auto"):
    """
    Convert onnx model into dnn_model model
    :param onnx_model: onnx model
    :param check_dnn_consistency check dnn_model consistency after loading the model
    :param data_layout: format of data tensors in  ["NHWC", "NCHW", "auto"]
    :param remove_skip: remove all skip-layers (e.g. Dropouts) from the result DNN
    :return: dnn_model as DNN class (see dnn_model.dnn_model.py)
    """
    try:
        if check_dnn_consistency:
            check_consistency(onnx_model)

        onnx_with_df = set_dataflow(onnx_model, check_dnn_consistency)
        if data_layout == "auto":
            data_layout = get_auto_data_layout(onnx_model, verbose=True)

        dnn = DNN()
        add_layers(onnx_with_df, dnn)
        all_output_names = find_all_output_names(onnx_model)
        add_connections(onnx_model, dnn)
        set_layers_dataflow(onnx_model, dnn, all_output_names, data_layout)

        clean_up_pure_data_layers(dnn)

        # print(all_output_names)
        return dnn
    except Exception as err:
        sys.stderr.write("ONNX to DNN conversion error")
        traceback.print_tb(err.__traceback__)
        return None


def clean_up_pure_data_layers(dnn):
    """
        Pure-data layers are layers in the DNN
        that only store data. In out DNN model
        such layers are defined as external I/Os or skipped
        at all!
    """
    for layer in dnn.get_layers():
        layer_input_connections = dnn.get_layer_input_connections(layer)
        if len(layer_input_connections) == 0 and layer.op in ["skip"]:
            # reset dataflow for dst nodes:
            # TODO: dirty trick, replace with pure-data nodes analysis!
            pure_data_output_connections = dnn.get_layer_output_connections(layer)
            for output_con in pure_data_output_connections:
                data_receiver = output_con.dst
                data_receiver_inputs = dnn.get_layer_input_connections(data_receiver)
                if len(data_receiver_inputs) > 1: # and data_receiver.op == "gemm"
                    for inp in data_receiver_inputs:
                        if inp.src != layer:
                            data_receiver.ifm = inp.src.ofm
                            data_receiver.ih = inp.src.oh
                            data_receiver.iw = inp.src.ow
            # remove pure data layer
            dnn.remove_layer(layer)


def get_auto_data_layout(onnx_model, verbose=True):
    """
    Determine data layout (NCHW or NHWC) for CNN layers
    The function assumes H==W for every input_examples tensor.
    Based on this information, the function detects the layout
    :return: data layout (NCHW or NHWC) for CNN layers
    """
    layout = "NCHW"
    try:
        model_inputs = get_onnx_model_inputs(onnx_model, verbose)
        if model_inputs:
            first_input = model_inputs[0]
            input_shape = extract_onnx_tensor_shape(first_input)
            if len(input_shape) == 4 or len(input_shape) == 3:
                # if two last dimensions match, it's NCHW
                if input_shape[-1] == input_shape[-2]:
                    layout = "NCHW"
                else:
                    layout = "NHWC"
            # print(input_shape, "HAS", layout, "layout")
    except Exception:
        if verbose:
            print("Warning: ONNX to DNN: auto-extraction of data layout have failed. Standard NCHW layout is used")
    return layout


def get_onnx_model_inputs(onnx_model, verbose):
    """
    Get ONNX model inputs
    :param onnx_model: onnx model
    :param verbose: verbose
    :return: onnx model inputs
    """
    inputs = []
    input_names, output_names = get_io_names(onnx_model, verbose=False)
    # print("input_examples names: ", input_names)
    if len(input_names) == 0:
        if verbose:
            print("WARNING: onnx parser: model has no feed (data) inputs")
    else:
        input_nodes = [node for node in onnx_model.graph.input]
        for inp in input_nodes:
            if inp.name in input_names:
                inputs.append(inp)

    return inputs


def get_onnx_model_outputs(onnx_model, verbose):
    """
    Get ONNX model outputs
    :param onnx_model: onnx model
    :param verbose: verbose
    :return: onnx model outputs
    """
    outputs = []
    input_names, output_names = get_io_names(onnx_model, verbose=False)
    # print("output names: ", output_names)
    if len(output_names) == 0:
        if verbose:
            print("WARNING: onnx parser: model has no data outputs")
    else:
        output_nodes = [node for node in onnx_model.graph.output]
        for outp in output_nodes:
            if outp.name in output_names:
                outputs.append(outp)
    return outputs


def get_io_names(onnx_model, verbose=False):
    """
    Determine input_examples and output nodes of a CNN
    """
    output = [node.name for node in onnx_model.graph.output]

    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))

    if verbose:
        print('Inputs: ', net_feed_input)
        print('Outputs: ', output)

    return net_feed_input, output


def check_consistency(model):
    """Check the model consistency"""
    onnx.checker.check_model(model)


def set_dataflow(model, check=True):
    """
    Set dataflow (input_examples and output tensor shapes) for every layer in .onnx DNN model
    :param model: onnx model with not-set input_examples flow
    :param check: check ONNX model consistency after setting onnx
    :return: onnx model with set input_examples flow
    """
    # Apply shape inference (input_examples formats computation) on the model
    inferred_model = shape_inference.infer_shapes(model)
    # Check the model
    if check:
        onnx.checker.check_model(inferred_model)
    # print('After shape inference, the shape info of Y is:\n{}'.format(inferred_model.graph.value_info))
    return inferred_model


def add_layers(onnx_model, dnn_model):
    """
    Parse layers of onnx model and add them to the dnn_model model
    :param onnx_model: onnx model
    :param dnn_model: dnn_model model
    """
    for node in onnx_model.graph.node:
        # print(node.name)
        op = onnx_to_dnn_op(node)
        layer = Layer(res=1, op=op, fs=1, ifm=1, ofm=1, bordermode="same")
        dnn_model.add_layer(layer)
        layer.name = node.name if node.name else layer.op + str(layer.id)
        layer.subop = onnx_to_dnn_subop(node)


def add_connections(onnx_model, dnn_model):
    """
    Add connections to a dnn_model model with layers
    :param onnx_model: onnx model
    :param dnn_model: corresponding dnn_model model
    """
    node_id = 0
    for node in onnx_model.graph.node:
        src_layer_id = node_id
        output_names = node.output if node.output is not None else []
        for output_name in output_names:
            # find input_examples owner
            dst_layer_ids = find_dst_layer_ids(onnx_model, output_name)
            for dst_layer_id in dst_layer_ids:
                dnn_model.connect_layers(src_layer_id, dst_layer_id)
        node_id = node_id + 1


def find_dst_layer_ids(onnx_model, input_name):
    """
    Find ids of all layers, which accept specified input_examples
    :param onnx_model: onnx model
    :param input_name: input_examples name
    :return: ids of all layers, which accept specified input_examples
    """
    layer_ids = []
    node_id = 0
    for node in onnx_model.graph.node:
        input_names = node.input if node.input is not None else []
        if input_name in input_names:
            layer_ids.append(node_id)
        node_id += 1
    return layer_ids


def find_all_output_names(onnx_model):
    """
    Find names of all tensors, referenced as layer outputs
    :param onnx_model: onnx model
    :return: names of all tensors, referenced as layer outputs
    """
    all_outputs = []
    for node in onnx_model.graph.node:
        output_names = node.output
        if output_names is not None:
            for output_name in output_names:
                all_outputs.append(output_name)
    return all_outputs


def set_layers_dataflow(onnx_model, dnn_model, all_output_names, data_layout):
    """
    Set dataflow of CNN layers
    :param onnx_model: onnx model
    :param dnn_model: dnn_model model
    :param all_output_names names of dnn_model layer outputs
    :param data_layout: tensor dims order (NCHW or NHWC)
    :return list of layer names (in layer traverse order)
    """
    node_id = 0
    layers = dnn_model.get_layers()
    model_input_names, model_output_names = get_io_names(onnx_model, False)

    for node in onnx_model.graph.node:
        layer = layers[node_id]

        if node_id == 0:
            set_input_tensor_for_input_layer(layer, onnx_model, data_layout)

        input_names = node.input if node.input is not None else []
        output_names = node.output if node.output is not None else []

        # print("input_examples: ", input_names, " len = ", len(input_names), "output: ", output_names, "len", len(output_names))

        # set output tensor (if exists)
        if len(output_names) > 0:
            output_tensor = find_onnx_data_tensor(onnx_model, output_names[0])
            set_output_tensor(layer, output_tensor, data_layout)

        # set input_examples tensor (if exists)
        if len(input_names) > 0:
            # layer has single input_examples
            if len(input_names) == 1:
                input_tensor = find_onnx_data_tensor(onnx_model, input_names[0])
                set_input_tensor(layer, input_tensor, data_layout)
            else:
                # layer has several inputs, among which are layer input_examples and layer parameters (weights and biases)
                for input_name in input_names:
                    # input_examples is an output of another tensor
                    if input_name in all_output_names or input_name in model_input_names:

                        input_tensor = find_onnx_data_tensor(onnx_model, input_name)
                        set_input_tensor(layer, input_tensor, data_layout)

        # set hyper-parameters
        # layer_param_inputs = [input_name for input_name in input_names if input_name not in all_output_names]
        set_kernel_size(layer, node)
        set_stride(layer, node)
        set_padding(layer, node)

        node_id = node_id + 1
        # input_tensor = find_onnx_data_tensor(onnx_model, input_names)
        # set_input_tensor(layer, input_tensor)


def set_input_tensor_for_input_layer(input_layer, onnx_model, data_layout):
    """
    Set input_examples tensor for the input_examples layer of CNN
    :param input_layer: input_examples layer of the CNN
    :param onnx_model: onnx model
    :param data_layout:  tensor dims order (NCHW or NHWC)
    """
    try:
        inputs = get_onnx_model_inputs(onnx_model, verbose=False)
        first_input_tensor = inputs[0]
        set_input_tensor(input_layer, first_input_tensor, data_layout)
    except Exception:
        print("WARNING: ONNX to DNN: I could not set input_examples data for input_examples layer")


def set_input_tensor(layer, onnx_tensor, data_layout):
    """
    Extract and set layer input_examples dims from input_examples tensor
    :param layer: DNN layer
    :param onnx_tensor: onnx tensor
    :param data_layout: tensor dims order (NCHW or NHWC)
    """
    try:
        shape = extract_onnx_tensor_shape(onnx_tensor)
        # print("input_examples shape for layer", layer.name, "is", shape)
        if data_layout == "NCHW":
            nchw_dims = tensor_shape_to_nchw(shape, onnx_tensor)
            i_n, i_c, i_h, i_w = nchw_dims
        else:
            nhwc_dims = tensor_shape_to_nhwc(shape, onnx_tensor)
            i_n, i_h, i_w, i_c = nhwc_dims

        layer.ifm = i_c
        layer.ih = i_h
        layer.iw = i_w
        layer.res = i_w
        # print("res: ", i_w)
        # print("set inp tensor ", nchw_dims, "to layer", layer.id)
    except Exception:
        print("ERROR: ONNX to DNN conversion. Layer", layer.name, "input_examples data setup error")


def set_output_tensor(layer, onnx_tensor, data_layout="NCHW"):
    """
    Extract and set layer input_examples dims from output tensor
    :param layer: DNN layer
    :param onnx_tensor: onnx tensor
    :param data_layout: tensor dims order (NCHW or NHWC)
    """
    shape = extract_onnx_tensor_shape(onnx_tensor)
    if shape is None:
        sys.stderr.write("ERROR: ONNX to DNN conversion. Layer " + layer.name +
                         " output data setup error: tensor shape is None\n")
        raise Exception("Output data setup error")

    if data_layout == "NCHW":
        nchw_dims = tensor_shape_to_nchw(shape, onnx_tensor)
        o_n, o_c, o_h, o_w = nchw_dims
    else:
        nhwc_dims = tensor_shape_to_nhwc(shape, onnx_tensor)
        o_n, o_h, o_w, o_c = nhwc_dims

    layer.ofm = o_c
    layer.oh = o_h
    layer.ow = o_w


def set_kernel_size(layer, onnx_node):
    """
    Set layer kernel size
    :param layer: layer
    :param onnx_node: corresponding onnx node
    """
    if layer.op not in ["conv", "pool"]:
        layer.fs = 1
        return

    default_fs = 3 if layer.op == "conv" else 2
    fs = extract_int_attribute_from_int_list_attributes(onnx_node, "kernel_shape")
    layer.fs = fs if fs > 0 else default_fs


def set_stride(layer, onnx_node):
    """
    Set layer kernel size
    :param layer: layer
    :param onnx_node: corresponding onnx node
    """
    if layer.op not in ["conv", "pool"]:
        layer.stride = 1
        return

    default_stride = 1 if layer.op == "conv" else 2
    stride = extract_int_attribute_from_int_list_attributes(onnx_node, "strides")

    if layer.op == "pool" and "global" in layer.subop:
        stride = layer.ow

    layer.stride = stride if stride > 0 else default_stride


def extract_int_attribute_from_int_list_attributes(onnx_node, name):
    """
    Extract hyper-parameters, represented as list of ints
    from onnx node
    :param onnx_node: onnx node
    :param name: name of the attribute
    :return: kernel size or -1 (if kernel size is not found)
    """
    not_found = -1
    try:
        for attribute in onnx_node.attribute:
            if attribute.name == name:
                first_dim = attribute.ints[0]
                if first_dim is not None:
                    return first_dim
    except Exception:
        pass
    return not_found


def set_padding(layer, onnx_node):
    """
    Set layer padding
    :param layer: layer
    :param onnx_node: corresponding onnx node
    """
    if layer.op not in ["conv", "pool"]:
        return

    padding = extract_int_list_attribute_from_int_list_attributes(onnx_node, "pads")

    if padding is None:
        set_auto_padding(layer)
    else:
        if len(padding) == 4:
            if is_uneven_padding(padding):
                # print(layer.name)
                padding = align_uneven_padding(padding)
            layer.pads = padding
        else:
            print("ONNX parser WARNING: I found but did not process padding of len", len(padding))


def is_uneven_padding(pads: [], verbose=False):
    """ Checks if padding is uneven"""
    if pads[0] != pads[2] or pads[1] != pads[3]: # or pads[0]!=pads[1]:
        if verbose:
            print("UNEVEN PADDING: ", pads)
        return True
    return False


def align_uneven_padding(pads):
    """ Uneven padding, i.e. padding where
    pads[0] != pads[2] or pads[1] != pads[3] is supported by ONNX,
    but is unsupported by most of the DL frameworks (including Keras and TensorRT)
    To avoid compatibility problems with DL frameworks, we align (transform) every
    uneven padding into the even padding
    """
    aligned_pads = [0, 0, 0, 0]
    sum_w_pad = pads[0] + pads[2]
    aligned_pads[0] = aligned_pads[2] = int(sum_w_pad/2 + sum_w_pad % 2)
    sum_h_pad = pads[1] + pads[3]
    aligned_pads[1] = aligned_pads[3] = int(sum_h_pad/2 + sum_h_pad % 2)
    # print("ALIGNED PADDING: ", pads, "-->", aligned_pads)
    return aligned_pads


def set_auto_padding(layer):
    """
    Set layer border mode (also referred as auto-padding)
    :param layer: layer
    """
    if layer.op not in ["conv", "pool"]:
        return

    if layer_has_null_or_empty_pads(layer):
        if layer.ih == layer.oh and layer.iw == layer.ow:
            layer.set_border_mode("same")


def extract_int_list_attribute_from_int_list_attributes(onnx_node, name):
    """
    Extract hyper-parameters, represented as list of ints
    from onnx node
    :param onnx_node: onnx node
    :param name: name of the attribute
    :return: kernel size or -1 (if kernel size is not found)
    """
    try:
        for attribute in onnx_node.attribute:
            if attribute.name == name:
                attr = []
                for attr_elem in attribute.ints:
                    attr.append(attr_elem)
                return attr
    except Exception:
        pass
    return None


def tensor_shape_to_nchw(tensor_shape, tensor):
    """
    Represent tensor shape as a [n, c, h, w] dims
    :param tensor_shape: tensor shape,
    :param tensor: onnx data tensor
    :return: tensor shape, aligned to 4D NCHW-format
    """
    if tensor is None:
        raise Exception("Tensor is None")

    if tensor_shape is None:
        raise Exception("data tensor shape is None")

    if len(tensor_shape) == 4:
        return tensor_shape

    if len(tensor_shape) == 3:
        n = 1
        c = tensor_shape[0]
        h = tensor_shape[1]
        w = tensor_shape[2]
        return [n, c, h, w]

    if len(tensor_shape) == 2:
        n = tensor_shape[0]
        c = tensor_shape[1]
        h = 1
        w = 1
        return [n, c, h, w]

    if len(tensor_shape) == 1:
        n = tensor_shape[0]
        c = 1
        h = 1
        w = 1
        return [n, c, h, w]

    if len(tensor_shape) == 0:
        return [0, 0, 0, 0]

    raise Exception("data tensor" + str(tensor) + " of shape " + str(tensor_shape) +
                    " has unexpected tensor len: " + str(len(tensor_shape)))


def tensor_shape_to_nhwc(tensor_shape, tensor):
    """
    Represent tensor shape as a [n, h, w,c] dims
    :param tensor_shape: tensor shape,
    :param tensor: onnx data tensor
    :return: tensor shape, aligned to 4D NHWC-format
    """
    if tensor_shape is None:
        return None
    if len(tensor_shape) == 4:
        return tensor_shape

    if len(tensor_shape) == 3:
        n = 1
        h = tensor_shape[0]
        w = tensor_shape[1]
        c = tensor_shape[2]
        return [n, h, w, c]

    if len(tensor_shape) == 2:
        n = 1
        c = tensor_shape[0]
        h = tensor_shape[1]
        w = 1
        return [n, c, h, w]

    if len(tensor_shape) == 1:
        n = 1
        c = tensor_shape[0]
        h = 1
        w = 1
        return [n, c, h, w]

    if len(tensor_shape) == 0:
        return [0, 0, 0, 0]

    raise Exception("data tensor" + str(tensor) + " of shape " + str(tensor_shape) +
                    " has unexpected tensor len: " + str(len(tensor_shape)))


def extract_onnx_tensor_shape(onnx_tensor):
    """
    Extract shape (list of dimension sizes) of onnx tensor
    :param onnx_tensor: onnx tensor
    :return: list of onnx tensor dimension sizes
    """
    if onnx_tensor is None:
        return None
    try:
        shape = onnx_tensor.type.tensor_type.shape
        shape_dims = []
        for dim in shape.dim:
            dim_value = dim.dim_value
            if dim_value > 0:
                shape_dims.append(dim_value)
        # print("I have extracted shape", shape_dims, "from tensor", onnx_tensor.name, "with shape", shape.dim)
        return shape_dims
    except Exception:
        try:
            shape = onnx_tensor.type.tensor_type.shape
            shape_dims = []
            for dim in shape.dim:
                dim_value = dim.dim_value
                if dim_value > 0:
                    shape_dims.append(dim_value)
            # print("I have extracted shape", shape_dims, "from tensor", onnx_tensor.name, "with shape", shape.dim)
            return shape_dims
        except Exception:
            return None


def onnx_to_dnn_subop(onnx_node):
    """
    Extract sub-operator (refined operator name) from onnx operator
    :param onnx_node: onnx node
    :return:
    """
    onnx_sub_op = onnx_node.op_type if onnx_node.op_type else "none"
    onnx_sub_op = onnx_sub_op.lower()
    return onnx_sub_op


def onnx_to_dnn_op(onnx_node) -> str:
    """
    Extract operator from onnx node description
    :param onnx_node: onnx node
    :return: DNN operator
    """
    onnx_op = onnx_node.op_type if onnx_node.op_type else "none"
    op = onnx_op
    if onnx_op.lower() in ["conv"]:
        op = "conv"
    if onnx_op.lower() in ["gemm", "fc", "matmul"]:
        op = "gemm"
    if onnx_op.lower() in ["maxpool", "averagepool", "globalaveragepool"]:
        op = "pool"
    if onnx_op.lower() in ["batchnormalization", "bn", "lrn"]:
        op = "normalization"
    if onnx_op.lower() in ["relu", "sigm", "leakyrelu"]:
        op = "activation"
    if onnx_op.lower() in ["softmax"]:
        op = "softmax"
    if onnx_op.lower() in ["add", "div", "mul", "sub"]:
        op = "arithmetic"
    # if onnx_op.lower() in ["flatten", "reshape"]:
    #    op = "reshape"
    if onnx_op.lower() == "concat":
        op = "concat"
    if onnx_op.lower() in ["dropout", "reshape", "flatten", "constant"]:
        op = "skip"
    return op


def find_onnx_data_tensor(onnx_model, tensor_name):
    """
    Parse layers of onnx model and add them to the dnn_model model
    :param onnx_model: onnx model
    :param tensor_name: name of onnx tensor
    """
    # search value info
    value_info = onnx_model.graph.value_info
    for vi in value_info:
        if vi.name == tensor_name:
            return vi

    # search model i/os
    model_inputs = get_onnx_model_inputs(onnx_model, verbose=False)
    for node in model_inputs:
        if node.name == tensor_name:
            return node

    model_outputs = get_onnx_model_outputs(onnx_model, verbose=False)
    for node in model_outputs:
        if node.name == tensor_name:
            return node
    # print("WARNING: tensor", tensor_name, "not found")
    """
    # search initializers
    initializers = [node for node in onnx_model.graph.initializer]
    for init in initializers:
        if init.name == tensor_name:
            print("I found init", str(init))
            return init

    # search nodes
    nodes = onnx_model.graph.node
    for node in nodes:
        if node.name == tensor_name:
            print("I found node", str(node))
            return node
    """

    return None




