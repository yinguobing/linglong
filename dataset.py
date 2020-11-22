"""
# Encoding labels

The raw labels, consisting of bounding boxes and class ids need to be
transformed into targets for training. This transformation consists of
the following steps:

- Generating anchor boxes for the given image dimensions
- Assigning ground truth boxes to the anchor boxes
- The anchor boxes that are not assigned any objects, are either assigned the
    background class or ignored depending on the IOU
- Generating the classification and regression targets using anchor boxes
"""

import tensorflow as tf
import numpy as np

from anchor import AnchorBox, compute_iou, swap_xy, convert_to_xywh, convert_to_corners
from preprocessing import random_flip_horizontal, resize_and_pad_image


class LabelEncoder:
    """Transforms the raw labels into targets for training.

    This class has operations to generate targets for a batch of samples which
    is made up of the input images, bounding boxes for the objects present and
    their class ids.

    Attributes:
      anchor_box: Anchor box generator to encode the bounding boxes.
      box_variance: The scaling factors used to scale the bounding box targets.
    """

    def __init__(self):
        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32)

    def _match_anchor_boxes(
            self, anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4):
        """Matches ground truth boxes to anchor boxes based on IOU.

        1. Calculates the pairwise IOU for the M `anchor_boxes` and N `gt_boxes`
          to get a `(M, N)` shaped matrix.
        2. The ground truth box with the maximum IOU in each row is assigned to
          the anchor box provided the IOU is greater than `match_iou`.
        3. If the maximum IOU in a row is less than `ignore_iou`, the anchor
          box is assigned with the background class.
        4. The remaining anchor boxes that do not have any class assigned are
          ignored during training.

        Arguments:
          anchor_boxes: A float tensor with the shape `(total_anchors, 4)`
            representing all the anchor boxes for a given input image shape,
            where each anchor box is of the format `[x, y, width, height]`.
          gt_boxes: A float tensor with shape `(num_objects, 4)` representing
            the ground truth boxes, where each box is of the format
            `[x, y, width, height]`.
          match_iou: A float value representing the minimum IOU threshold for
            determining if a ground truth box can be assigned to an anchor box.
          ignore_iou: A float value representing the IOU threshold under which
            an anchor box is assigned to the background class.

        Returns:
          matched_gt_idx: Index of the matched object
          positive_mask: A mask for anchor boxes that have been assigned ground
            truth boxes.
          ignore_mask: A mask for anchor boxes that need to by ignored during
            training
        """
        iou_matrix = compute_iou(anchor_boxes, gt_boxes)
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(max_iou, match_iou)
        negative_mask = tf.less(max_iou, ignore_iou)
        ignore_mask = tf.logical_not(
            tf.logical_or(positive_mask, negative_mask))
        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=tf.float32),
            tf.cast(ignore_mask, dtype=tf.float32),
        )

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        """Transforms the ground truth boxes into targets for training"""
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] -
                 anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ],
            axis=-1,
        )
        box_target = box_target / self._box_variance
        return box_target

    def _encode_sample(self, image_shape, gt_boxes, cls_ids):
        """Creates box and classification targets for a single sample"""
        anchor_boxes = self._anchor_box.get_anchors(
            image_shape[1], image_shape[2])
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(
            anchor_boxes, gt_boxes
        )
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)
        cls_target = tf.where(
            tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids
        )
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        cls_target = tf.expand_dims(cls_target, axis=-1)
        label = tf.concat([box_target, cls_target], axis=-1)
        return label

    def encode_batch(self, batch_images, gt_boxes, cls_ids):
        """Creates box and classification targets for a batch"""
        images_shape = tf.shape(batch_images)
        batch_size = images_shape[0]

        labels = tf.TensorArray(
            dtype=tf.float32, size=batch_size, dynamic_size=True)
        for i in range(batch_size):
            label = self._encode_sample(images_shape, gt_boxes[i], cls_ids[i])
            labels = labels.write(i, label)
        batch_images = tf.keras.applications.resnet.preprocess_input(
            batch_images)
        return batch_images, labels.stack()


def parse_record(example_proto):
    feature_description = {
        'image/height': tf.io.FixedLenFeature((), tf.int64),
        'image/width': tf.io.FixedLenFeature((), tf.int64),
        'image/filename': tf.io.FixedLenFeature((), tf.string),
        'image/encoded': tf.io.FixedLenFeature((), tf.string),
        'image/format': tf.io.FixedLenFeature((), tf.string),
        'faces/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'faces/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'faces/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'faces/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'faces/label': tf.io.VarLenFeature(tf.int64),
    }

    example = tf.io.parse_single_example(example_proto, feature_description)

    for k in example:
        if isinstance(example[k], tf.SparseTensor):
            if example[k].dtype == tf.string:
                example[k] = tf.sparse.to_dense(example[k], default_value='')
            else:
                example[k] = tf.sparse.to_dense(example[k], default_value=0)

    return example


def preprocess_data(sample):
    """Applies preprocessing step to a single sample

    Arguments:
      sample: A dict representing a single training sample.

    Returns:
      image: Resized and padded image with random horizontal flipping applied.
      bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
        of the format `[x, y, width, height]`.
      class_id: An tensor representing the class id of the objects, having
        shape `(num_objects,)`.
    """

    image = tf.image.decode_jpeg(sample['image/encoded'], channels=3)
    bbox = tf.stack([sample["faces/bbox/ymin"],
                     sample["faces/bbox/xmin"],
                     sample["faces/bbox/ymax"],
                     sample["faces/bbox/xmax"]], axis=-1)

    bbox = swap_xy(bbox)
    class_id = tf.cast(sample["faces/label"], dtype=tf.int32)

    image, bbox = random_flip_horizontal(image, bbox)
    image, image_shape, _ = resize_and_pad_image(image)

    bbox = tf.stack([bbox[:, 0] * image_shape[1],
                     bbox[:, 1] * image_shape[0],
                     bbox[:, 2] * image_shape[1],
                     bbox[:, 3] * image_shape[0]],
                    axis=-1)
    bbox = convert_to_xywh(bbox)
    return image, bbox, class_id


def load_datasets(batch_size):

    train_dataset = tf.data.TFRecordDataset(
        "/home/robin/data/face/wider/tfrecord/wider_train.record")
    val_dataset = tf.data.TFRecordDataset(
        "/home/robin/data/face/wider/tfrecord/wider_val.record")

    # Setting up a `tf.data` pipeline

    # To ensure that the model is fed with data efficiently we will be using
    # `tf.data` API to create our input pipeline. The input pipeline
    # consists for the following major processing steps:

    # - Apply the preprocessing function to the samples
    # - Create batches with fixed batch size. Since images in the batch can
    # have different dimensions, and can also have different number of
    # objects, we use `padded_batch` to the add the necessary padding to create
    # rectangular tensors
    # - Create targets for each sample in the batch using `LabelEncoder`

    autotune = tf.data.experimental.AUTOTUNE

    label_encoder = LabelEncoder()

    train_dataset = train_dataset.map(parse_record,
                                      num_parallel_calls=autotune)
    train_dataset = train_dataset.map(preprocess_data,
                                      num_parallel_calls=autotune)
    train_dataset = train_dataset.shuffle(8 * batch_size)
    train_dataset = train_dataset.padded_batch(batch_size=batch_size,
                                               padding_values=(0.0, 1e-8, -1),
                                               drop_remainder=True)
    train_dataset = train_dataset.map(label_encoder.encode_batch,
                                      num_parallel_calls=autotune)
    train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
    train_dataset = train_dataset.prefetch(autotune)

    val_dataset = val_dataset.map(parse_record, num_parallel_calls=autotune)
    val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
    val_dataset = val_dataset.padded_batch(batch_size=1,
                                           padding_values=(0.0, 1e-8, -1),
                                           drop_remainder=True)
    val_dataset = val_dataset.map(label_encoder.encode_batch,
                                  num_parallel_calls=autotune)
    val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
    val_dataset = val_dataset.prefetch(autotune)

    return train_dataset, val_dataset


if __name__ == "__main__":
    import cv2
    val_dataset = tf.data.TFRecordDataset(
        "/home/robin/data/face/wider/tfrecord/wider_val.record")
    val_dataset = val_dataset.map(parse_record)
    val_dataset = val_dataset.map(preprocess_data)

    for image, boxes, class_id in val_dataset:

        # Use OpenCV to preview the image.
        image = np.array(image, np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width, _ = image.shape

        for cx, cy, w, h in boxes:
            cv2.rectangle(image,
                          (int(cx - w/2), int(cy - h/2)),
                          (int(cx + w/2), int(cy + h/2)),
                          (0, 255, 0), 2)

        # Show the result
        cv2.imshow("image", image)
        if cv2.waitKey() == 27:
            break
