
def convert_onnx_to_keras(onnx_model, input_name):
    """ Convert an onnx model into Keras model for training
    :param onnx_model ONNX model
    :param input_name name of the input_examples node in the ONNX model
    """
    from onnx2keras import onnx_to_keras
    import onnx
    k_model = onnx_to_keras(onnx_model, [input_name])
    return k_model


def convert_test(model_path="/home/svetlana/ONNX/Dolly/app_mnist/SCN1.onnx", input_name='input_data'):
    """
    Test onnx-to-keras conversion
    :param model_path:
    :param input_name: name of the input_examples node in the ONNX model
    :return:
    """
    k_model = convert_onnx_to_keras(model_path, input_name)
    print(k_model)
    return k_model

# convert_test()