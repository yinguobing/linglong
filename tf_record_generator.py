import tensorflow as tf
import os
import numpy as np
import cv2
from tqdm import tqdm


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    """Returns a float_list from a float / double list."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    """Returns an int64_list from a bool / enum / int / uint list."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


class DetectionSample(object):

    def __init__(self, image_file, boxes):
        """Construct an object detection sample.

        Args:
            image: image file path.
            boxes: numpy array of bounding boxes [[x, y, w, h], ...]

        """
        self.image_file = image_file
        self.boxes = boxes

    def read_image(self, format="BGR"):
        """Read in image as numpy array in format of BGR by defult, else RGB.

        Args:
            format: the channel order, default BGR.

        Returns:
            a numpy array.
        """
        img = cv2.imread(self.image_file)
        if format != "BGR":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img


class WiderFace(object):

    def __init__(self, dataset_path, mode="train"):
        """Construct the WiderFace dataset.

        Dataset file structure:

            wider  <-- path to here
            ├── wider_face_split
            │   ├── wider_face_train_bbx_gt.txt
            │   ├── wider_face_train.mat
            │   ├── ...
            ├── WIDER_train
            │   ├── 0--Parade
            │   ├── 10--People_Marching
            │   ├── ...
            └── WIDER_val
                ├── 0--Parade
                ├── 10--People_Marching
                ├── ...


        Args:
            dataset_path: path to the dataset directory.
        """
        # Find the label files.
        label_file_train = os.path.join(
            dataset_path, "wider_face_split", "wider_face_train_bbx_gt.txt")
        label_file_val = os.path.join(
            dataset_path, "wider_face_split", "wider_face_val_bbx_gt.txt")
        label_file_test = os.path.join(
            dataset_path, "wider_face_split", "wider_face_test_filelist.txt")

        # Parse the label files to get image file path and bounding boxes.
        def _parse(label_file, img_dir):
            samples = []
            with open(label_file, "r") as fid:
                while(True):
                    # Find out which image file to be processed.
                    line = fid.readline()
                    if line == "":
                        break
                    line = line.rstrip('\n').rstrip()
                    assert line.endswith(".jpg"), "Failed to read next label."
                    img_file = os.path.join(dataset_path, img_dir, line)

                    # Read the bounding boxes.
                    n_boxes = int(fid.readline().rstrip('\n').rstrip())
                    if n_boxes == 0:
                        fid.readline()
                        continue
                    lines = [fid.readline().rstrip('\n').rstrip().split(' ')
                             for _ in range(n_boxes)]

                    boxes = np.array(lines, dtype=np.float32)[:, :4]

                    # Accumulate the results.
                    samples.append(DetectionSample(img_file, boxes))
            return samples

        if mode == 'train':
            self.dataset = _parse(label_file_train, "WIDER_train")
        elif mode == 'val':
            self.dataset = _parse(label_file_val, "WIDER_val")
        elif mode == 'test':
            # There is no bounding boxes in test dataset.
            self.dataset = []
            with open(label_file_test, "r") as fid:
                for img_file in fid:
                    self.dataset.append(DetectionSample(img_file, None))
        else:
            raise ValueError(
                'Mode {} not supported, check again.'.format(mode))

        # Set index for iterator.
        self.index = 0

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == len(self.dataset):
            raise StopIteration
        sample = self.dataset[self.index]
        self.index += 1
        return sample


def create_tf_example(example, min_size=None):

    img = example.read_image()
    height, width, _ = img.shape

    # Filename of the image.
    filename = example.image_file.split('/')[-1].encode("utf-8")

    # Encoded image bytes
    with tf.io.gfile.GFile(example.image_file, 'rb') as fid:
        encoded_image_data = fid.read()

    image_format = example.image_file.split('.')[-1].encode("utf-8")

    # Transform the bbox size.
    boxes_wider = example.boxes
    xmin, ymin, wbox, hbox = np.split(boxes_wider, 4, axis=1)

    # Filter boxes whose size exceeds the threshold.
    if min_size:
        mask = np.all((wbox > min_size, hbox > min_size))

        # Incase all boxes are invalid.
        if not np.any(mask):
            print("No valid face box found.")
            return None

        xmin = xmin[mask]
        ymin = ymin[mask]
        wbox = wbox[mask]
        hbox = hbox[mask]

    # In case the coordinates are flipped.
    xs = np.concatenate((xmin, xmin+wbox), axis=-1)
    ys = np.concatenate((ymin, ymin+hbox), axis=-1)

    xmin = np.min(xs, axis=-1)
    xmax = np.max(xs, axis=-1)
    ymin = np.min(ys, axis=-1)
    ymax = np.max(ys, axis=-1)

    # Make sure all boxes are in image boundaries.
    ymax = (np.clip(ymax, a_min=0, a_max=height).flatten() / height).tolist()
    xmax = (np.clip(xmax, a_min=0, a_max=width).flatten() / width).tolist()
    ymin = (np.clip(ymin, a_min=0, a_max=height).flatten() / height).tolist()
    xmin = (np.clip(xmin, a_min=0, a_max=width).flatten() / width).tolist()

    # List of integer class id of bounding box (1 per box)
    classes = [1 for _ in xmin]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
        'image/encoded': bytes_feature(encoded_image_data),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/ymin': float_list_feature(ymin),
        'image/object/bbox/xmin': float_list_feature(xmin),
        'image/object/bbox/ymax': float_list_feature(ymax),
        'image/object/bbox/xmax': float_list_feature(xmax),
        'image/object/class/label': int64_list_feature(classes),
        'image/object/class/text': bytes_feature('face'.encode('utf8'))
    }))

    return tf_example


if __name__ == "__main__":
    writer = tf.io.TFRecordWriter(
        "/home/robin/data/face/wider/tfrecord/wider_train.record")

    # Read in your dataset to examples variable
    data_dir = "/home/robin/data/face/wider"
    wider = WiderFace(data_dir, mode="train")

    for example in tqdm(wider):
        tf_example = create_tf_example(example, 24)
        if tf_example is not None:
            writer.write(tf_example.SerializeToString())

    writer.close()
