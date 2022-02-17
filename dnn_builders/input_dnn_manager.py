"""
Builds keras model from input_examples, specified as:
1) a path to existing model in one of the supported input_examples formats or
2) name of a CNN from keras models zoo, supported by this tool
"""

# import sub-modules
import support_matrix

#################################################################
# DNN in internal DNN format for analysis and system-level design


def load_or_build_dnn_for_analysis(dnn_spec, dnn_name=None, verbose=True):
    """
    Load dnn from file or build it from keras library
    :param dnn_spec: dnn specified as:
    1) name of a dnn from keras DL framework library (e.g. vgg16) or
    2) path to dnn with supported extension (e.g. vgg16.onnx or vgg16.h5)
    :param dnn_name dnn name
    :param verbose: print details
    :return: dnn, represented as internal dnn model, defined in models
    """
    from support_matrix import supported_dnn_extensions, supported_keras_models

    is_file_path = is_dnn_a_supported_file_path(dnn_spec)
    is_prebuilt = is_dnn_a_supported_prebuilt_model(dnn_spec)

    if is_file_path is False and is_prebuilt is False:
        print("DNN model ", dnn_spec, "cannot be built. Please, choose a prebuilt model from")
        print("keras: ", supported_keras_models())
        print("or explicitly specify model extension within", supported_dnn_extensions())

    # get dnn model source
    dnn_name = "dnn" if dnn_name is None else dnn_name
    if is_prebuilt:
        dnn_name = dnn_spec
    dnn_src = get_dnn_source(dnn_spec, dnn_name, is_file_path, is_prebuilt)

    # get dnn model
    analysis_dnn = create_or_load_analysis_dnn_model(dnn_spec, dnn_src, verbose)
    analysis_dnn.name = dnn_name
    return analysis_dnn


def create_or_load_analysis_dnn_model(dnn_model_spec,
                                      dnn_model_src,
                                      verbose=True):
    """
    Create dnn model or load it from file
    :param dnn_model_spec: dnn specified as:
    1) name of a dnn form keras/custom library (e.g. vgg16) or
    2) path to dnn with extension (e.g. vgg16.onnx or vgg16.h5)
    :param dnn_model_src: source of the dnn model
    :param verbose: print details
    :return: dnn model, created or loaded from file
    """
    from dnn_builders.keras_models_builder import build_keras_model
    from dnn_builders.test_analysis_dnn_builder import create_test_analysis_dnn_model
    from converters.onnx_to_dnn import read_and_check_consistency, onnx_to_dnn
    from converters.keras_to_dnn import keras_to_dnn

    if dnn_model_src == "onnx":
        stage = "ONNX model loading"
        if verbose:
            print(stage, verbose)
        onnx_model = read_and_check_consistency(dnn_model_spec, verbose=verbose)
        stage = "ONNX -> Analysis dnn model conversion"
        if verbose:
            print(stage, verbose)
        dnn_model = onnx_to_dnn(onnx_model)
        return dnn_model

    if dnn_model_src == "h5":
        stage = "Keras .h5 model loading"
        if verbose:
            print(stage, verbose)
        from fileworkers.keras_fw import load_keras_model
        dnn_keras = load_keras_model(dnn_model_spec)
        stage = "Keras -> Analysis dnn model conversion"
        if verbose:
            print(stage, verbose)
        dnn_model = keras_to_dnn(dnn_keras, data_layout="auto")
        return dnn_model

    if dnn_model_src == "keras" or dnn_model_src == "tf_zoo":
        stage = "Keras model " + dnn_model_spec + " creation"
        if verbose:
            print(stage, verbose)
        dnn_keras = build_keras_model(dnn_model_spec)
        stage = "Keras -> Analysis dnn model conversion"
        if verbose:
            print(stage, verbose)
        dnn_model = keras_to_dnn(dnn_keras, data_layout="auto")
        return dnn_model

    if dnn_model_src == "analysis":
        stage = "Test analysis dnn model " + dnn_model_spec + " creation"
        if verbose:
            print(stage, verbose)
        dnn_model = create_test_analysis_dnn_model(dnn_model_spec)
        return dnn_model

    raise Exception("unsupported input_examples dnn model source: " + dnn_model_src)

#################################################
# DNN in Keras format for training and validation


def load_or_build_dnn_for_training(dnn_spec, conf):
    """
    Load keras dnn from file or build it from keras library
    :param dnn_spec: dnn specified as:
    1) name of a dnn form keras library (e.g. vgg16) or
    2) path to dnn with extension (e.g. vgg16.onnx or vgg16.h5)
    :param conf execution config
    :return: dnn, represented as keras dnn model, suitable for training and validation
    """
    from support_matrix import supported_dnn_extensions, supported_keras_models, supported_custom_dnns

    is_file_path = is_dnn_a_supported_file_path(dnn_spec)
    is_prebuilt = is_dnn_a_supported_prebuilt_model(dnn_spec)

    if is_file_path is False and is_prebuilt is False:
        print("DNN model ", dnn_spec, "cannot be built. Please, choose a prebuilt model from")
        print("keras: ", supported_keras_models(), "custom: ", supported_custom_dnns())
        print("or explicitly specify model extension within", supported_dnn_extensions())

    # get dnn model source
    dnn_name = "dnn" if conf is None else conf.dnn_model
    dnn_src = get_dnn_source(dnn_spec, dnn_name, is_file_path, is_prebuilt)

    input_shape = None
    classes = None

    if conf is not None:
        input_shape, classes, data_layout = extract_data_param_from_conf(conf)
        import tensorflow.keras.backend as keras_backend
        keras_backend.set_image_data_format(data_layout)

    # get dnn model
    keras_dnn = create_or_load_keras_dnn_model(dnn_spec, dnn_src,
                                               input_shape, classes,
                                               conf.freeze_baseline,
                                               conf.reset_dense_weights,
                                               conf.verbose)

    return keras_dnn


def extract_data_param_from_conf(conf):
    """
    Extract data parameters from execution config
    :param conf:  execution config
    :return: data parameters, extracted from execution config
    """
    from util import data_layout_from_ds_name, channels_num_from_ds_name, get_input_shape, classes_num_from_ds_name
    # get dnn model parameters
    # data_layout = "channels_last"
    data_layout = data_layout_from_ds_name(conf.dataset)
    # TODO: make more generic
    if conf.preprocessor == "pytorch":
        data_layout = "channels_first"

    from tensorflow.python.keras import backend as K
    K.set_image_data_format(data_layout)
    im_channels = channels_num_from_ds_name(conf.dataset)

    # dnn_model I/Os configuration - make manually configurable!
    input_shape = get_input_shape(data_layout, conf.img_resolution, im_channels)
    classes = classes_num_from_ds_name(conf.dataset)
    return input_shape, classes, data_layout


def create_or_load_keras_dnn_model(dnn_model_spec,
                                   dnn_model_src,
                                   input_shape,
                                   classes,
                                   freeze_baseline,
                                   reset_dense_weights,
                                   verbose=True):
    """
    Create dnn model or load it from file
    :param dnn_model_spec: dnn specified as:
    1) name of a dnn form keras/custom library (e.g. vgg16) or
    2) path to dnn with extension (e.g. vgg16.onnx or vgg16.h5)
    :param dnn_model_src: source of the dnn model
    :param input_shape: input_examples shape
    :param classes: number of classes in the model
    :param freeze_baseline: freeze pretrained baseline (for keras models pretrained on ImageNet)
    :param reset_dense_weights: re-initialize classifier (for keras models pretrained on ImageNet)
    :param verbose: if True, details of model creation or loading will be printed
    :return: dnn model, created or loaded from file
    """
    from dnn_builders.keras_models_builder import build_keras_model
    from dnn_builders.custom_dnn_builder import create_custom_dnn_model

    if dnn_model_src == "onnx":
        stage = "ONNX model loading"
        if verbose:
            print(stage, verbose)
        onnx_model = load_onnx_model(dnn_model_spec)
        stage = "ONNX -> Keras dnn model conversion"
        if verbose:
            print(stage, verbose)
        dnn_model = convert_onnx_model_to_keras(onnx_model, False)
        return dnn_model

    if dnn_model_src == "h5":
        stage = "Keras .h5 model loading"
        if verbose:
            print(stage, verbose)
        from fileworkers.keras_fw import load_keras_model
        dnn_keras = load_keras_model(dnn_model_spec)
        return dnn_keras

    if dnn_model_src == "keras" or dnn_model_src == "tf_zoo":
        stage = "Keras model " + dnn_model_spec + " creation"
        if verbose:
            print(stage, verbose)
        dnn_model = build_keras_model(dnn_model_spec, input_shape, classes,
                                      reset_dense_weights, freeze_baseline)
        return dnn_model

    if dnn_model_src == "custom":
        stage = "Custom dnn model " + dnn_model_spec + " creation"
        if verbose:
            print(stage, verbose)
        dnn_model = create_custom_dnn_model(dnn_model_spec, input_shape, classes)
        return dnn_model

    raise Exception("unknown dnn model source: " + dnn_model_src)


def load_onnx_model(filepath):
    """
    Load onnx model from file
    :param filepath: path of the file
    :return: dnn model_loaded from file
    """
    import onnx
    loaded_onnx_model = onnx.load(filepath)
    return loaded_onnx_model


def convert_onnx_model_to_keras(onnx_model, verbose):
    """
    Convert onnx model to keras dnn model
    :param onnx_model: onnx model
    :param verbose: verbose
    :return: keras dnn model
    """
    from onnx2keras import onnx_to_keras
    model = onnx_to_keras(onnx_model, ['input_data'],
                          verbose=verbose, name_policy='renumerate')
    return model


def is_dnn_a_supported_file_path(dnn):
    """
    Check if dnn is a file path
    :param dnn: dnn
    :return: True if dnn is a file path and false otherwise
    """
    from support_matrix import supported_dnn_extensions
    for extension in supported_dnn_extensions():
        if dnn.endswith(extension):
            return True
    return False


def is_dnn_a_supported_prebuilt_model(dnn):
    """
    Check if dnn is a supported pre-built model
    :param dnn: dnn
    :return: True if dnn is a a supported pre-built model and false otherwise
    """
    from support_matrix import supported_keras_models, supported_custom_dnns, supported_test_analysis_dnns
    return dnn in supported_keras_models() or dnn in supported_custom_dnns() or dnn in supported_test_analysis_dnns()


def get_dnn_source(dnn, dnn_name, is_file, is_prebuilt):
    """
    Get source of dnn model
    :param dnn: dnn specified as:
    1) name of a dnn form keras library (e.g. vgg16) or
    2) path to dnn with extension (e.g. vgg16.onnx or vgg16.h5)
    :param dnn_name: name of the dnn model
    :param is_file: is dnn a supported file?
    :param is_prebuilt: is dnn a prebuilt model?
    :return: source of dnn model
    """
    if is_file:
        ext = get_supported_dnn_file_extension(dnn)
        ext = ext.replace(".", "")
        return ext
    if is_prebuilt:
        if dnn_name in support_matrix.supported_keras_models():
            return "keras"
        if dnn_name in support_matrix.supported_custom_dnns():
            return "custom"
        if dnn_name in support_matrix.supported_test_analysis_dnns():
            return "analysis"
    return "unknown"


def get_supported_dnn_file_extension(dnn_path):
    """
    Get extension of a dnn model, specified as a path to a file
    :param dnn_path: dnn
    :return: True if dnn is a file path and false otherwise
    """
    from support_matrix import supported_dnn_extensions
    for extension in supported_dnn_extensions():
        if dnn_path.endswith(extension):
            return extension
    return False

