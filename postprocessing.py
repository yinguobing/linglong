import tensorflow as tf

from anchor import AnchorBox, convert_to_corners


def build_decoding_layer(num_classes=2,
                         confidence_threshold=0.05,
                         nms_iou_threshold=0.5,
                         max_detections_per_class=100,
                         max_detections=100,
                         box_variance=[0.1, 0.1, 0.2, 0.2],):
    """Decodes predictions of the RetinaNet model.

    Args:
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
    _anchor_box = AnchorBox()
    _box_variance = tf.convert_to_tensor(
        [0.1, 0.1, 0.2, 0.2], dtype=tf.float32)

    def _decode_box_predictions(anchor_boxes, box_predictions):
        boxes = box_predictions * _box_variance
        boxes = tf.concat([boxes[:, :, :2] * anchor_boxes[:, :, 2:] +
                           anchor_boxes[:, :, :2],
                           tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:]],
                          axis=-1)
        boxes_transformed = convert_to_corners(boxes)

        return boxes_transformed

    def forward(images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = _anchor_box.get_anchors(image_shape[1], image_shape[2])
        box_predictions = predictions[:, :, :4]
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
        boxes = _decode_box_predictions(
            anchor_boxes[None, ...], box_predictions)

        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            max_detections_per_class,
            max_detections,
            nms_iou_threshold,
            confidence_threshold,
            clip_boxes=False)

    return forward
