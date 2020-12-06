"""
# Building the ResNet50 backbone

RetinaNet uses a ResNet based backbone, using which a feature pyramid network
is constructed. In the example we use ResNet50 as the backbone, and return the
feature maps at strides 8, 16 and 32.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from models.shufflenet_v2 import shuffle_unit_v2


def shuffle_net_v2(input_shape, filters=116):
    # Build conv layers.
    stage2_layers = [shuffle_unit_v2(downsampling=True, filters=filters)]
    stage2_layers.extend([shuffle_unit_v2() for _ in range(3)])

    stage3_layers = [shuffle_unit_v2(downsampling=True)]
    stage3_layers.extend([shuffle_unit_v2() for _ in range(7)])

    stage4_layers = [shuffle_unit_v2(downsampling=True)]
    stage4_layers.extend([shuffle_unit_v2() for _ in range(3)])

    # Inputs defined here.
    inputs = keras.Input(input_shape, dtype=tf.float32)

    # Forward propgation and get feature map for stage 2, 3, and 4.
    x = keras.layers.Conv2D(24, 3, 2, 'same')(inputs)
    x = keras.layers.MaxPool2D(3, 2, 'same')(x)

    for layer in stage2_layers:
        x = layer(x)
    x_2 = x

    for layer in stage3_layers:
        x = layer(x)
    x_3 = x

    for layer in stage4_layers:
        x = layer(x)
    x_4 = x

    # Output the feature maps.
    outputs = [x_2, x_3, x_4]

    # Finally, build the model.
    model = keras.Model(inputs=inputs, outputs=outputs, name='shuffle_net_v2')

    return model


def get_backbone():
    return shuffle_net_v2((None, None, 3))


def feature_pyramid(backbone=None):
    """Builds the Feature Pyramid with the feature maps from the backbone.

    Args:
        backbone: The backbone to build the feature pyramid from.
    """

    _backbone = backbone if backbone else get_backbone()

    conv_s2_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
    conv_s2_3x3 = keras.layers.Conv2D(256, 3, 1, "same")

    conv_s3_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
    conv_s3_3x3 = keras.layers.Conv2D(256, 3, 1, "same")

    conv_s4_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
    conv_s4_3x3 = keras.layers.Conv2D(256, 3, 1, "same")

    conv_s5_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
    conv_s6_3x3 = keras.layers.Conv2D(256, 3, 2, "same")

    upsample_2x = keras.layers.UpSampling2D(2)

    def forward(inputs):
        x_2, x_3, x_4 = _backbone(inputs)

        # Match channels.
        p_2 = conv_s2_1x1(x_2)
        p_3 = conv_s3_1x1(x_3)
        p_4 = conv_s4_1x1(x_4)

        # Up sampling.
        p_3 = p_3 + upsample_2x(p_4)
        p_2 = p_2 + upsample_2x(p_3)

        # Merge
        p_2 = conv_s2_3x3(x_2)
        p_3 = conv_s3_3x3(p_3)
        p_4 = conv_s4_3x3(p_4)

        # Additional layers for FPN
        p_5 = conv_s5_3x3(x_4)
        p_6 = conv_s6_3x3(tf.nn.relu(p_5))

        return [p_2, p_3, p_4, p_5, p_6]

    return forward


def build_head(output_filters, bias_init):
    """Builds the class/box predictions head.

    The RetinaNet model has separate heads for bounding box regression and
    for predicting class probabilities for the objects. These heads are shared
    between all the feature maps of the feature pyramid.

    Arguments:
        output_filters: Number of convolution filters in the final layer.
        bias_init: Bias Initializer for the final convolution layer.

    Returns:
        A keras sequential model representing either the classification
            or the box regression head depending on `output_filters`.
    """
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    head = []

    for _ in range(4):
        head.append(keras.layers.Conv2D(256, 3, padding="same",
                                        kernel_initializer=kernel_init))
        head.append(keras.layers.ReLU())

    head.append(keras.layers.Conv2D(output_filters, 3, 1, padding="same",
                                    kernel_initializer=kernel_init,
                                    bias_initializer=bias_init))

    def forward(inputs):
        for layer in head:
            inputs = layer(inputs)

        return inputs

    return forward


def build_retinanet(num_classes, backbone=None):
    """Implementing the RetinaNet architecture.

    Args:
        num_classes: Number of classes in the dataset.
        backbone: The backbone to build the feature pyramid from.
    """
    fpn = feature_pyramid(backbone)
    prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
    cls_head = build_head(9 * num_classes, prior_probability)
    box_head = build_head(9 * 4, "zeros")

    inputs = keras.Input((None, None, 3), dtype=tf.float32)
    batch_size = tf.shape(inputs)[0]
    cls_outputs = []
    box_outputs = []
    features = fpn(inputs)

    for feature in features:
        box_outputs.append(tf.reshape(box_head(feature),
                                      [batch_size, -1, 4]))
        cls_outputs.append(tf.reshape(cls_head(feature),
                                      [batch_size, -1, num_classes]))

    cls_outputs = tf.concat(cls_outputs, axis=1)
    box_outputs = tf.concat(box_outputs, axis=1)

    outputs = tf.concat([box_outputs, cls_outputs], axis=-1)

    return keras.Model(inputs=inputs, outputs=outputs, name="retinanet")


def build_model(num_classes):
    backbone = get_backbone()
    return build_retinanet(num_classes, backbone)
