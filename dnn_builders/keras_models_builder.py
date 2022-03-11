"""
Builds keras models
"""


def supported_keras_models():
    return ["vgg16", "resnet50",
            "mobilenetv1", "mobilenetv1_25", "mobilenetv2", "mobilenetv2_25",
            "efficientnetb0", "densenet121", "inceptionv2"]# "resnet18


def build_keras_model(model_name, input_shape=None, classes=1000, custom_classifier=False, freeze_baseline=True):
    import tensorflow
    import tensorflow.keras.applications
    """
    Load one of the supported Keras models (models from Keras.applications Zoo)
    Args:
        model_name: model name
        input_shape: input_examples data shape
        classes: number of classed (ofm in output Dense layer)
        custom_classifier: use custom classifier
        freeze_baseline: freeze baseline model (for pretrained models)

    Returns: dnn model
    """

    model = None
    include_top = not custom_classifier

    if model_name == "vgg16":
        from tensorflow.keras.applications.vgg16 import VGG16
        if custom_classifier is True:
            model = VGG16(include_top=include_top, weights='imagenet',
                          input_shape=input_shape, classes=classes,
                          pooling="max")
            if freeze_baseline:
                freeze_weights(model)
            model = add_custom_classifier(model, classes)
        else:
            model = VGG16(include_top=include_top, weights='imagenet')
            if freeze_baseline:
                freeze_weights(model)

    if model_name == "resnet18":
        from dnn_builders.renset_builder import ResNet18
        from tensorflow import keras
        if custom_classifier is True:
            base_model = ResNet18(input_shape, weights='imagenet', classes=classes, include_top=include_top)
            x = keras.layers.GlobalAveragePooling2D()(base_model.output)
            model = keras.models.Model(inputs=[base_model.input], outputs=[x])
            if freeze_baseline:
                freeze_weights(model)
            # model = add_custom_classifier(model, classes, hidden_dense_neurons=[256, 128], dropout_rates=[0.2, 0.1])
            model = add_custom_classifier(model, classes, hidden_dense_neurons=[256, 128], dropout_rates=[0, 0])
        else:
            model = ResNet18(weights='imagenet', classes=classes, include_top=include_top)
            if freeze_baseline:
                freeze_weights(model)

    if model_name == "resnet50":
        from tensorflow.keras.applications.resnet50 import ResNet50
        if custom_classifier is True:
            model = ResNet50(include_top=include_top, weights='imagenet',
                             input_shape=input_shape, classes=classes,
                             pooling="avg")
            if freeze_baseline:
                freeze_weights(model)
            model = add_custom_classifier(model, classes, hidden_dense_neurons=[256], dropout_rates=[0.1])
            # model = add_custom_classifier(model, classes,
            #                              hidden_dense_neurons=[4000, 2000, 1000, 500],
            #                              dropout_rates=[0, 0.4, 0.3, 0.2])
        else:
            model = ResNet50(include_top=include_top, weights='imagenet')
            if freeze_baseline:
                freeze_weights(model)

    if model_name == "mobilenetv1":
        from tensorflow.keras.applications.mobilenet import MobileNet
        if custom_classifier is True:
            model = MobileNet(include_top=include_top, weights='imagenet',
                                input_shape=input_shape, classes=classes,
                                pooling="max")
            if freeze_baseline:
                freeze_weights(model)
            model = add_custom_classifier(model, classes, hidden_dense_neurons=[256], dropout_rates=[0.1])
        else:
            model = MobileNet(include_top=include_top, weights='imagenet')
            if freeze_baseline:
                freeze_weights(model)

    if model_name == "mobilenetv1_25":
        from tensorflow.keras.applications.mobilenet import MobileNet
        if custom_classifier is True:
            model = MobileNet(include_top=include_top, weights='imagenet',
                              input_shape=input_shape, classes=classes,
                              pooling="max", alpha=0.25)
            if freeze_baseline:
                freeze_weights(model)
            model = add_custom_classifier(model, classes, hidden_dense_neurons=[256], dropout_rates=[0.1])
        else:
            model = MobileNet(include_top=include_top, weights='imagenet')
            if freeze_baseline:
                freeze_weights(model)

    if model_name == "mobilenetv2":
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
        if custom_classifier is True:
            model = MobileNetV2(include_top=include_top, weights='imagenet',
                                input_shape=input_shape, classes=classes,
                                pooling="max")
            if freeze_baseline:
                freeze_weights(model)
            model = add_custom_classifier(model, classes, hidden_dense_neurons=[256], dropout_rates=[0.1])
        else:
            model = MobileNetV2(include_top=include_top, weights='imagenet')
            if freeze_baseline:
                freeze_weights(model)

    if model_name == "mobilenetv2_25":
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
        if custom_classifier is True:
            model = MobileNetV2(include_top=include_top, weights='imagenet',
                                input_shape=input_shape, classes=classes,
                                pooling="max", alpha=0.25)
            if freeze_baseline:
                freeze_weights(model)
            model = add_custom_classifier(model, classes, hidden_dense_neurons=[256], dropout_rates=[0.1])
        else:
            model = MobileNetV2(include_top=include_top, weights='imagenet')
            if freeze_baseline:
                freeze_weights(model)

    if model_name == "efficientnetb0":
        from tensorflow.keras.applications.efficientnet import EfficientNetB0
        if custom_classifier is True:
            model = EfficientNetB0(include_top=include_top, weights='imagenet',
                                input_shape=input_shape, classes=classes,
                                pooling="max")
            if freeze_baseline:
                freeze_weights(model)
            model = add_custom_classifier(model, classes, hidden_dense_neurons=[256, 128], dropout_rates=[0.1, 0.1])
        else:
            model = EfficientNetB0(include_top=include_top, weights='imagenet')
            if freeze_baseline:
                freeze_weights(model)

    if model_name == "densenet121":
        from tensorflow.keras.applications.densenet import DenseNet121
        if custom_classifier is True:
            model = DenseNet121(include_top=include_top, weights='imagenet',
                                input_shape=input_shape, classes=classes,
                                pooling="max")
            if freeze_baseline:
                freeze_weights(model)
            model = add_custom_classifier(model, classes, hidden_dense_neurons=[256], dropout_rates=[0.1])
        else:
            model = DenseNet121(include_top=include_top, weights='imagenet')
            if freeze_baseline:
                freeze_weights(model)

    if model_name == "inceptionv2":
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
        if custom_classifier is True:
            model = InceptionResNetV2(include_top=include_top, weights='imagenet',
                                      input_shape=input_shape, classes=classes,
                                      pooling="max")
            if freeze_baseline:
                freeze_weights(model)
            model = add_custom_classifier(model, classes, hidden_dense_neurons=[256], dropout_rates=[0.1])
        else:
            model = InceptionResNetV2(include_top=include_top, weights='imagenet')
            if freeze_baseline:
                freeze_weights(model)

    if model is None:
        raise Exception("unknown keras model", model_name)

    print("keras model", model_name, "loaded with")
    print("  - custom classifier = ", custom_classifier, ", i.e., include_top = ", include_top)
    print("  - frozen (not trainable) baseline:", freeze_baseline)
    return model


def freeze_weights(keras_model):
    """
    Freeze (i.e. prevent from training) weights in a dnn
    :param keras_model: keras dnn model
    """
    for layer in keras_model.layers:
        layer.trainable = False

    # for layer in base_model.layers: if
    # isinstance(layer, BatchNormalization): layer.trainable = True else: layer.trainable = False


def freeze_all_weights_but_dense(keras_model):
    """
    Freeze (i.e. prevent from training) weights in a dnn
    :param keras_model: keras dnn model
    """
    import tensorflow

    for layer in keras_model.layers:
        if not isinstance(layer, tensorflow.keras.layers.Dense):
            layer.trainable = False

    # for layer in base_model.layers: if
    # isinstance(layer, BatchNormalization): layer.trainable = True else: layer.trainable = False


def add_custom_classifier(base_model, classes, hidden_dense_neurons=[1024], dropout_rates=None):
    """
    Add custom classification part
    Args:
        base_model: base CNN model (feature extractor only)
        classes: number of classes (number of neurons in the output layer of the classifier)
        hidden_dense_neurons: number of neurons in hidden dense layers of classifier
        dropout_rates: dropout rates for hidden dense layers of classifier
    Returns: CNN model with added custom classifier

    """
    import tensorflow

    drop_rates = dropout_rates
    if drop_rates is None:
        drop_rates = []
        for hdn in hidden_dense_neurons:
            drop_rates.append(0)

    # construct the head of the model that will be placed on top of the
    # the base model (classifier)
    head_model = base_model.output

    head_model = tensorflow.keras.layers.Flatten()(head_model)

    for hidden_dense_id in range(len(hidden_dense_neurons)):
        hdn = hidden_dense_neurons[hidden_dense_id]
        head_model = tensorflow.keras.layers.Dense(hdn, activation="relu")(head_model)
        dropout_rate = dropout_rates[hidden_dense_id]
        if dropout_rate > 0:
            head_model = tensorflow.keras.layers.Dropout(dropout_rate)(head_model)

    head_model = tensorflow.keras.layers.Dense(classes, activation='softmax')(head_model) # activation="softmax"
    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    new_model = tensorflow.keras.Model(inputs=base_model.input, outputs=head_model)
    # model.add(tf.keras.layers.Dense(classes))
    return new_model



