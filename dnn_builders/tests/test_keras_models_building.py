from dnn_builders_zoo.keras_models_builder import build_keras_model


def test_resnet50_building():
    # --------------------------------------------
    model_name = "resnet50"
    classes = 10
    input_shape = (224, 224, 3)# (32, 32, 3)
    custom_classifier = True
    freeze_baseline = True

    print("expected model with model_name =", model_name, ", input_shape =", input_shape, ", classes=", classes,
          "custom_classifier=", custom_classifier, ", freeze_baseline=", freeze_baseline)

    keras_model = build_keras_model(model_name, input_shape, classes,
                                    custom_classifier=custom_classifier, freeze_baseline=freeze_baseline)
    keras_model.summary()
    # --------------------------------------------



test_resnet50_building()