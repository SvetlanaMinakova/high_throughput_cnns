from models.dnn_model.dnn import DNN, Layer
import tensorflow as tf


def keras_to_dnn(keras_model, data_layout): # ="NCHW"
    """
    Convert keras model into dnn_model model
    :param keras_model: keras model
    :param data_layout: tensor dims order (NCHW or NHWC)
    :return: dnn_model as DNN class (see dnn_model.dnn_model.py)
    """

    # keras_model.summary()

    if data_layout in ["AUTO", "auto"]:
        data_layout = extract_data_layout(keras_model)

    dnn = DNN()
    add_layers(keras_model, dnn, data_layout)
    add_connections(keras_model, dnn)

    return dnn


def extract_data_layout(keras_model):
    input_shape = keras_model.input_shape
    if len(input_shape) == 4:
        if input_shape[-1] == input_shape[-2]:
            return "NCHW"
        else:
            return "NHWC"
    print("WARNING: I could not extract data format. Default NCHW format is returned")
    return "NCHW"


def add_connections(keras_model, dnn):
    # intermediate_data_tensors = get_intermediate_data_tensors(keras_model)
    # print("i've got data tensors!")
    for layer in keras_model.layers:
        add_layer_connections(keras_model, layer, dnn)


def add_layer_connections(keras_model, keras_layer, dnn):
    """
    Add connections between the layers
    :param keras_model keras model
    :param keras_layer: keras layer
    :param dnn dnn
    """
    layer_inp = keras_layer.input

    if isinstance(layer_inp, tf.Tensor):
        layer_inputs_list = [layer_inp]

    else:
        # multi-input_examples layers
        if isinstance(layer_inp, list):
            layer_inputs_list = [li for li in layer_inp]
        # one-input_examples layers
        else:
            layer_inputs_list = [layer_inp]

    for inp_tensor in layer_inputs_list:
        for producer in keras_model.layers:
            producer_out = producer.output
            if inp_tensor.ref() == producer_out.ref():
                # print("layer", keras_layer.name, "accepts input_examples from layer", producer.name)

                dst_layer_name = keras_layer.name
                src_layer_name = producer.name

                src_layer = dnn.find_layer_by_name(src_layer_name)
                dst_layer = dnn.find_layer_by_name(dst_layer_name)

                # print("connect src", src_layer, "with dst", dst_layer)
                if src_layer is not None and dst_layer is not None:
                    dnn.connect_layers(src_layer.id, dst_layer.id)


def add_layers(keras_model, dnn_model, data_layout):
    """
    Parse layers of onnx model and add them to the dnn_model model
    :param keras_model: onnx model
    :param dnn_model: dnn_model model
    :param data_layout: tensor dims order (NCHW or NHWC)
    """
    for keras_layer in keras_model.layers:
        layer = create_dnn_layer(keras_layer, data_layout)
        if not layer.subop == "inputlayer":
            dnn_model.add_layer(layer)


def create_dnn_layer(keras_layer, data_layout):
    """
    From a keras layer create a dnn layer
    :param keras_layer: keras layer
    :param data_layout: tensor dims order (NCHW or NHWC)
    :return:
    """

    # determine operator
    op = keras_to_dnn_op(keras_layer)
    sub_op = keras_to_dnn_sub_op(keras_layer)

    # determine hyperparametets from I/O formats
    if data_layout == "NCHW":
        n_in, c_in, h_in, w_in = keras_tensor_shape_nchw(keras_layer.input)
        n_out, c_out, h_out, w_out = keras_tensor_shape_nchw(keras_layer.output)

        # print(keras_layer.name, "data layout:", data_layout,
        #      "in: ", [n_in, c_in, h_in, w_in],
        #      ", out: ", [n_out, c_out, h_out, w_out])

    else:
        # print(keras_layer.name, "input: ", keras_layer.input)
        #NHWC
        n_in, h_in, w_in, c_in = keras_tensor_shape_nhwc(keras_layer.input)
        n_out, h_out, w_out, c_out = keras_tensor_shape_nhwc(keras_layer.output)

        # print(keras_layer.name, "data layout:", data_layout,
        #      "in: ", [n_in, c_in, h_in, w_in],
        #      ", out: ", [n_out, c_out, h_out, w_out])

    fs = extract_filter_size(keras_layer, op, sub_op)
    stride = extract_stride(keras_layer, op, sub_op)
    bordermode = extract_bordermode(op, w_in, w_out)

    # create layer
    # print("create layer with res=", w_in, ", op=", op, "fs=", fs, "ifm=", c_in, ", ofm=", c_out, "bordermode=", bordermode)
    layer = Layer(res=w_in, op=op, fs=fs, ifm=c_in, ofm=c_out, bordermode=bordermode)
    layer.stride = stride
    layer.oh = h_out
    layer.ow = w_out
    layer.subop = sub_op

    # set name
    layer.name = keras_layer.name if keras_layer.name else layer.name

    ##########################
    # process padding

    # process implicit padding
    if bordermode == "same":
        layer.set_autopads()

    # process explicit padding
    if layer.subop == "padding":
        wpad1 = int(max((layer.ow - layer.iw)/2, 0))
        wpad2 = int(max((layer.ow - layer.iw - wpad1), 0))
        hpad1 = int(max((layer.oh - layer.ih)/2, 0))
        hpad2 = int(max((layer.oh - layer.ih - hpad1), 0))
        pads = [wpad1, wpad2, hpad1, hpad2]
        layer.pads = pads

    return layer


def keras_tensor_shape_nchw(keras_tensor):
    """
    Extract shape of keras tensor in nchw format
    :param keras_tensor keras data tensor
    :return: dnn tensor in preferred data layout
    """

    if keras_tensor is None:
        return None
    # TODO: check!
    # multi-input_examples layers
    if isinstance(keras_tensor, list):
        keras_tensor = keras_tensor[0]

    tensor_shape = keras_tensor.shape

    if len(tensor_shape) == 4:
            n = tensor_shape[0] if tensor_shape[0] is not None else 1
            c = tensor_shape[1]
            h = tensor_shape[2]
            w = tensor_shape[3]
            return [n, c, h, w]

    if len(tensor_shape) == 3:
            c = tensor_shape[0]
            n = c
            h = tensor_shape[1]
            w = tensor_shape[2]
            return [n, c, h, w]

    if len(tensor_shape) == 2:
        n = tensor_shape[0] if tensor_shape[0] is not None else 1
        c = tensor_shape[1]
        h = 1
        w = 1
        return [n, c, h, w]


def keras_tensor_shape_nhwc(keras_tensor):
    """
    Extract shape of keras tensor in nchw format
    :param keras_tensor keras data tensor
    :return: dnn tensor in preferred data layout
    """

    if keras_tensor is None:
        return None
    # TODO: check!
    # multi-input_examples layers
    if isinstance(keras_tensor, list):
        # print(keras_tensor)
        # print()
        keras_tensor = keras_tensor[0]


    tensor_shape = keras_tensor.shape

    if len(tensor_shape) == 4:
            n = tensor_shape[0] if tensor_shape[0] is not None else 1
            h = tensor_shape[1]
            w = tensor_shape[2]
            c = tensor_shape[3]
            return [n, h, w, c]

    if len(tensor_shape) == 3:
            c = tensor_shape[3]
            n = c
            h = tensor_shape[2]
            w = tensor_shape[1]
            return [n, h, w, c]

    if len(tensor_shape) == 2:
        n = tensor_shape[0] if tensor_shape[0] is not None else 1
        c = tensor_shape[1]
        h = 1
        w = 1
        return [n, h, w, c]


def keras_to_dnn_op(keras_layer) -> str:
    """
    Extract operator from keras layer description
    :param keras_layer: keras layer
    :return: DNN operator
    """
    keras_op = keras_layer.__class__.__name__ if keras_layer.__class__.__name__ else "none"
    op = keras_op
    if keras_op.lower() in ["conv2d", "conv3d", "depthwiseconv2d", "depthwiseconv3d"]:
        op = "conv"
    if keras_op.lower() in ["gemm", "fc", "matmul", "dense"]:
        op = "gemm"
    if keras_op.lower() in ["maxpool", "averagepool", "maxpooling2d", "averagepooling2d", "globalaveragepool", "globalaveragepooling2d"]:
        op = "pool"
    if keras_op.lower() in ["batchnormalization", "bn", "lrn", "normalization"]:
        op = "normalization"
    if keras_op.lower() in ["activation", "relu"]:
        op = "activation"
    if keras_op.lower() in ["add", "div", "mul", "multiply", "sub"]:
        op = "arithmetic"
    if keras_op.lower() in ["flatten", "reshape"]:
        op = "reshape"
    if keras_op.lower() in ["concat", "concatenate"]:
        op = "concat"
    if keras_op.lower() in ["dropout", "reshape", "flatten", "inputlayer"]:
        op = "skip"
    if keras_op.lower().startswith("zeropadding"):
        op = "skip"
    return op


def keras_to_dnn_sub_op(keras_layer) -> str:
    """
    Extract operator from keras layer description
    :param keras_layer: keras layer
    :return: DNN operator
    """
    keras_op = keras_layer.__class__.__name__ if keras_layer.__class__.__name__ else "none"
    sub_op = keras_op.lower()

    if keras_op.lower() in ["conv2d", "conv3d"]:
        sub_op = "conv"

    if keras_op.lower() in ["depthwiseconv2d", "depthwiseconv3d"]:
        sub_op = "depthwiseconv"

    if keras_op.lower() in ["gemm", "fc", "matmul", "dense"]:
        sub_op = "gemm"

    # pooling types
    if keras_op.lower() == "maxpooling2d":
        sub_op = "maxpool"

    if keras_op.lower() == "averagepooling2d":
        sub_op = "averagepool"

    if keras_op.lower() == "globalaveragepooling2d":
        sub_op = "globalaveragepool"

    if keras_op.lower() in ["batchnormalization", "bn"]:
        sub_op = "batchnormalization"

    if keras_op.lower().startswith("zeropadding"):
        sub_op = "padding"

    if keras_op.lower() in ["mul", "multiply"]:
        sub_op = "mul"

    if keras_op.lower() in ["add"]:
        sub_op = "add"

    return sub_op


def extract_filter_size(keras_layer, dnn_op, dnn_subop):
    fs = 1
    if dnn_op in ["conv"]:
        fs = keras_layer.kernel_size[0]
    if dnn_op in ["pool"] and not dnn_subop.startswith("global"):
        fs = keras_layer.pool_size[0]
    return fs


def extract_stride(keras_layer, dnn_op, dnn_subop):
    stride = 1
    if dnn_op in ["conv", "pool"]:
        if not dnn_subop.startswith("global"):
            stride = keras_layer.strides[0]
    return stride


def extract_bordermode(dnn_op, w_in, w_out):
    bm = "same"
    if dnn_op in ["conv", "pool"]:
         if w_in > w_out:
             bm = "valid"
    return bm


def print_ops_per_layer(keras_model):
    for layer in keras_model.layers:
        print(layer.__class__.__name__)