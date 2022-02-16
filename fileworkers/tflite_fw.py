import tensorflow as tf
import os
from support_matrix import quantization_options


def save_as_tf_model(keras_model, filepath="model.tflite", representative_data_gen=None,
                     replace=False, quantization=None, verbose=False, verify=True):
    """
    Save keras model as a tflite model file
    :param keras_model: keras model
    :param filepath: path to output tflite model
    :param quantization: level of quantization in [None, 'default', 'float16', 'int']
    :param representative_data_gen representative data generator for int-quantization
    :param replace: replace existing file with quantized model
    :param verbose: verbose
    :param verify: verify that the quantized model is saved
    """
    if replace is False and os.path.isfile(filepath):
        if verbose:
            print("TFlite (quantized) model already exists in ", filepath, ". Saving skipped.")
        return

    if quantization is None or quantization == "none":
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        tflite_model = converter.convert()  # saving converted model in "converted_model.tflite" file

    else:
        qo = quantization_options()
        if quantization not in qo:
            print("WARNING: No quantization applied to a keras model. Unknown quantization option", quantization,
                  ". Please choose from", qo)

        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        if quantization == "default":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if quantization == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        if quantization == "int":
            if representative_data_gen is None:
                print("WARNING: No quantization applied to a keras model. "
                      "Int-quantization requires non-empty representative data generator",
                      ". Please provide the data generator for int-quantization", qo)
            else:
                converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
                converter.representative_dataset = representative_data_gen

        tflite_model = converter.convert()

    open(filepath, "wb").write(tflite_model)

    if verify:
        verify_quant_model_saved(tflite_model, verbose)

    if verbose:
        print("TFlite model saved at", filepath)


def verify_quant_model_saved(model_path, verbose):
    """
    Verify that the quantized model was successfully saved
    :param model_path: path to quantized model
    :param verbose: print details
    :return:
    """
    try:
        tf_interpreter = load_tf_model(model_path, verbose=verbose)
        return True
    except Exception as err:
        print(err)
        # print(traceback.format_exc())
        return False


def load_tf_model(path, verbose=False):
    """
    Load tflite model from path
    :param path: path
    :param verbose: verbose
    :return: TFlite model interpreter
    """
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()  # Get input_examples and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()  # Test model on some input_examples data.
    input_shape = input_details[0]['shape']

    if verbose:
        print("TFlite model loaded from", path)
    return interpreter

