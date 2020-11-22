"""
# Building the ResNet50 backbone

RetinaNet uses a ResNet based backbone, using which a feature pyramid network
is constructed. In the example we use ResNet50 as the backbone, and return the
feature maps at strides 8, 16 and 32.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from models.blazenet import blaze_block, double_blaze_block


def blaze_stem(filters):
    stem_layers = [
        keras.layers.Conv2D(filters=filters,
                            kernel_size=(5, 5),
                            strides=(2, 2),
                            padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu')]

    def forward(inputs):
        for layer in stem_layers:
            inputs = layer(inputs)

        return inputs

    return forward


def blaze_net(input_shape):
    # Stacking layers for 16x16 feature maps.
    blaze_layers_16x16 = [
        blaze_stem(24),
        blaze_block(24),
        blaze_block(24),
        blaze_block(48, (2, 2)),
        blaze_block(48),
        blaze_block(48),
        double_blaze_block(24, 96, (2, 2)),
        double_blaze_block(24, 96),
        double_blaze_block(24, 96)]

    # Stacking layers for 8x8 feature maps.
    blaze_layers_8x8 = [
        double_blaze_block(24, 96, (2, 2)),
        double_blaze_block(24, 96),
        double_blaze_block(24, 96)]

    # Inputs defined here.
    inputs = keras.Input((input_shape), dtype=tf.float32)

    # Forward propgation till feature map 16x16 got.
    x = inputs
    for layer in blaze_layers_16x16:
        x = layer(x)
    x_16 = x

    # Keep moving forward till feature map 8x8 got.
    for layer in blaze_layers_8x8:
        x = layer(x)
    x_8 = x

    outputs = [x_16, x_8]

    # Finally, build the model.
    model = keras.Model(inputs=inputs, outputs=outputs, name='blaze_net')

    return model


def get_backbone():
    return blaze_net((128, 128, 3))


# Building Feature Pyramid Network as a custom layer

class FeaturePyramid(keras.layers.Layer):
    """Builds the Feature Pyramid with the feature maps from the backbone.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, backbone=None, **kwargs):
        super(FeaturePyramid, self).__init__(name="FeaturePyramid", **kwargs)
        self.backbone = backbone if backbone else get_backbone()
        self.conv_c3_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c3_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = keras.layers.Conv2D(256, 3, 1, "same")

        self.upsample_2x = keras.layers.UpSampling2D(2)

    def call(self, images, training=False):
        f_16, f_8 = self.backbone(images)

        # Match channels
        p_16 = self.conv_c3_1x1(f_16)
        p_8 = self.conv_c4_1x1(f_8)

        p_16 = p_16 + self.upsample_2x(p_8)

        p_16 = self.conv_c3_3x3(p_16)
        p_8 = self.conv_c4_3x3(p_8)

        return [p_16, p_8]


"""
## Building the classification and box regression heads.

The RetinaNet model has separate heads for bounding box regression and
for predicting class probabilities for the objects. These heads are shared
between all the feature maps of the feature pyramid.
"""


def build_head(output_filters, bias_init):
    """Builds the class/box predictions head.

    Arguments:
      output_filters: Number of convolution filters in the final layer.
      bias_init: Bias Initializer for the final convolution layer.

    Returns:
      A keras sequential model representing either the classification
        or the box regression head depending on `output_filters`.
    """
    head = keras.Sequential([keras.Input(shape=[None, None, 256])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(4):
        head.add(
            keras.layers.Conv2D(256, 3, padding="same",
                                kernel_initializer=kernel_init))
        head.add(keras.layers.ReLU())
    head.add(
        keras.layers.Conv2D(output_filters, 3, 1, padding="same",
                            kernel_initializer=kernel_init,
                            bias_initializer=bias_init,))

    return head


# Building RetinaNet using a subclassed model

class RetinaNet(keras.Model):
    """A subclassed Keras model implementing the RetinaNet architecture.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, num_classes, backbone=None, **kwargs):
        super(RetinaNet, self).__init__(name="RetinaNet", **kwargs)
        self.fpn = FeaturePyramid(backbone)
        self.num_classes = num_classes

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.cls_head = build_head(9 * num_classes, prior_probability)
        self.box_head = build_head(9 * 4, "zeros")

    def call(self, image, training=False):
        features = self.fpn(image, training=training)
        N = tf.shape(image)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(
                tf.reshape(self.cls_head(feature), [N, -1, self.num_classes])
            )
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        return tf.concat([box_outputs, cls_outputs], axis=-1)


# Implementing a custom layer to decode predictions

class DecodePredictions(tf.keras.layers.Layer):
    """A Keras layer that decodes predictions of the RetinaNet model.

    Attributes:
      num_classes: Number of classes in the dataset
      confidence_threshold: Minimum class probability, below which detections
        are pruned.
      nms_iou_threshold: IOU threshold for the NMS operation
      max_detections_per_class: Maximum number of detections to retain per
       class.
      max_detections: Maximum number of detections to retain across all
        classes.
      box_variance: The scaling factors used to scale the bounding box
        predictions.
    """

    def __init__(
        self,
        num_classes=80,
        confidence_threshold=0.05,
        nms_iou_threshold=0.5,
        max_detections_per_class=100,
        max_detections=100,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        **kwargs
    ):
        super(DecodePredictions, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] +
                anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners(boxes)
        return boxes_transformed

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_box.get_anchors(
            image_shape[1], image_shape[2])
        box_predictions = predictions[:, :, :4]
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
        boxes = self._decode_box_predictions(
            anchor_boxes[None, ...], box_predictions)

        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )


def build_model(num_classes):
    backbone = get_backbone()
    return RetinaNet(num_classes, backbone)
