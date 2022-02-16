
def save_as_onnx(keras_model, output_path="model.onnx", name="model"):
    import keras2onnx
    """ Convert an onnx model into Keras model for training
    :param output_path path to result ONNX model
    :param input_name name of the input_examples node in the ONNX model
    """
    # convert to onnx model
    onnx_model = keras2onnx.convert_keras(keras_model, name)
    keras2onnx.save_model(onnx_model, output_path)