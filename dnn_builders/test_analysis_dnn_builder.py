from models.dnn_model.dnn import DNN, Layer


def supported_test_analysis_dnns():
    return ['testDNN', 'testDNNConcat', 'testDNNResidual']


def create_test_analysis_dnn_model(model_name):
    """
    Build a test analysis dnn model
    Args:
        model_name: model name

    Returns: dnn model

    """

    if model_name == "testDNN":
        return get_test_dnn_simple()
    if model_name == "testDNNConcat":
        return get_test_dnn_concat()
    if model_name == "testDNNResidual":
        return get_test_dnn_residual()

    raise Exception("unknown test analysis dnn model: " + model_name)


def get_test_dnn_simple():
    dnn = DNN("testDNN")
    conv1 = Layer(32, "conv", 3, 3, 8, "valid")
    conv1.oh = conv1.ow = 30

    relu1 = Layer(30, "activation", 1, 8, 8, "same")
    relu1.oh = conv1.ow = 30
    relu1.subop = "relu"

    max_pool1 = Layer(conv1.oh, "pool", 2, 8, 8, "valid")
    max_pool1.subop = "maxpool"
    max_pool1.stride = 2
    max_pool1.oh = max_pool1.ow = 15

    fc1 = Layer(max_pool1.ow, "gemm", 1, max_pool1.ofm, 10, "same")
    fc1.oh = fc1.ow = 1

    dnn.stack_layer(conv1)
    dnn.stack_layer(relu1)
    dnn.stack_layer(max_pool1)
    dnn.stack_layer(fc1)

    dnn.set_auto_ios()

    return dnn


def get_test_dnn_max_mem_save1():
    # create DNN
    dnn = DNN("CNN1")

    # create layers
    inp_data = Layer(32, "data", 1, 1, 3, "valid")
    inp_data.oh = inp_data.ow = 32
    inp_data.name = "l1_1(input)"
    inp_data.subop = "input"

    conv1 = Layer(32, "conv", 5, 3, 8, "same")
    conv1.oh = conv1.ow = 32
    conv1.name = "l2_1(conv)"

    conv2 = Layer(32, "conv", 5, 8, 8, "same")
    conv2.oh = conv2.ow = 32
    conv2.name = "l3_1(conv)"

    add = Layer(32, "arithmetic", 1, 16, 16, "valid")
    add.oh = conv1.ow = 32
    add.subop = "add"
    add.name = "l4_1(add)"

    outp_data = Layer(32, "relu", 1, 16, 16, "valid")
    outp_data.oh = outp_data.ow = 32
    outp_data.name = "l5_1(output)"
    outp_data.subop = "output"

    # add layers and connections
    dnn.stack_layer(inp_data)
    dnn.stack_layer(conv1)
    dnn.stack_layer(conv2)
    dnn.stack_layer(add)
    dnn.stack_layer(outp_data)

    # add residual connection
    dnn.connect_layers_by_name("l2_1(conv)", "l4_1(add)")
    # dnn.set_auto_ios()

    return dnn


def get_test_dnn_max_mem_save2():
    # create DNN
    dnn = DNN("CNN2")

    # create layers
    inp_data = Layer(32, "data", 1, 1, 3, "valid")
    inp_data.oh = inp_data.ow = 32
    inp_data.name = "l1_2(input)"
    inp_data.subop = "input"

    conv1 = Layer(32, "conv", 5, 3, 8, "valid")
    conv1.oh = conv1.ow = 28
    conv1.name = "l2_2(conv)"

    fc = Layer(28, "gemm", 1, 8, 10, "valid")
    fc.oh = fc.ow = 1
    fc.subop = "gemm"
    fc.name = "l3_2(gemm)"

    outp_data = Layer(1, "data", 1, 10, 10, "valid")
    outp_data.oh = outp_data.ow = 1
    outp_data.name = "l4_2(output)"
    outp_data.subop = "output"

    # add layers and connections
    dnn.stack_layer(inp_data)
    dnn.stack_layer(conv1)
    dnn.stack_layer(fc)
    dnn.stack_layer(outp_data)
    # dnn.set_auto_ios()

    return dnn


def get_test_dnn_concat():
    dnn = DNN("testDNNConcat")
    conv1 = Layer(32, "conv", 3, 3, 8, "valid")
    conv1.oh = conv1.ow = 30
    conv1.name = "conv1"

    max_pool1 = Layer(conv1.oh, "pool", 2, 8, 8, "valid")
    max_pool1.subop = "maxpool"
    max_pool1.stride = 2
    max_pool1.oh = max_pool1.ow = 15
    max_pool1.name = "maxpool1"

    conv2 = Layer(conv1.oh, "conv", 2, 8, 8, "same")
    conv2.stride = 2
    conv2.oh = max_pool1.ow = 15
    conv2.name = "conv2"

    relu2 = Layer(conv2.oh, "relu", 2, 8, 8, "same")
    relu2.op = "activation"
    relu2.subop = "relu"
    relu2.oh = relu2.ow = 15
    relu2.name = "relu2"

    concat = Layer(conv2.oh, "concat", fs=1,
                   ifm=(max_pool1.ofm + conv2.ofm),
                   ofm=(max_pool1.ofm + conv2.ofm),
                   bordermode="same")
    concat.oh = concat.ow = concat.res
    concat.name = "concat"

    fc1 = Layer(max_pool1.ow, "gemm", 1, max_pool1.ofm, 10, "same")
    fc1.oh = fc1.ow = 1

    dnn.stack_layer(conv1)
    dnn.stack_layer(max_pool1)
    dnn.add_layer(conv2)
    dnn.stack_layer(relu2)
    dnn.connect_layers_by_name("conv1", "conv2")
    dnn.add_layer(concat)
    dnn.connect_layers_by_name("maxpool1", "concat")
    dnn.connect_layers_by_name("relu2", "concat")
    dnn.stack_layer(fc1)

    dnn.set_auto_ios()

    return dnn


def get_test_dnn_residual():
    dnn = DNN("testDNNResidual")
    conv1 = Layer(32, "conv", 3, 3, 8, "valid")
    conv1.oh = conv1.ow = 30
    conv1.name = "conv1"

    max_pool1 = Layer(conv1.oh, "pool", 2, 8, 8, "valid")
    max_pool1.subop = "maxpool"
    max_pool1.stride = 2
    max_pool1.oh = max_pool1.ow = 15
    max_pool1.name = "maxpool1"

    conv2 = Layer(conv1.oh, "conv", 2, 8, 8, "same")
    conv2.stride = 1
    conv2.oh = max_pool1.ow = 15
    conv2.name = "conv2"

    relu2 = Layer(conv2.oh, "relu", 2, 8, 8, "same")
    relu2.op = "activation"
    relu2.subop = "relu"
    relu2.oh = relu2.ow = 15
    relu2.name = "relu2"

    add = Layer(max_pool1.oh, "arithmetic", fs=1,
                ifm=max_pool1.ofm,
                ofm=max_pool1.ofm,
                bordermode="same")
    add.oh = add.ow = add.res
    add.name = add.subop = "add"

    fc1 = Layer(max_pool1.ow, "gemm", 1, max_pool1.ofm, 10, "same")
    fc1.oh = fc1.ow = 1

    dnn.stack_layer(conv1)
    dnn.stack_layer(max_pool1)
    dnn.stack_layer(conv2)
    dnn.stack_layer(relu2)
    dnn.stack_layer(add)
    # add residual connection
    dnn.connect_layers_by_name("maxpool1", "add")
    dnn.stack_layer(fc1)
    dnn.set_auto_ios()

    return dnn




