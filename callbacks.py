import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from postprocessing import build_decoding_layer
from preprocessing import resize_and_pad_image


class LogImages(keras.callbacks.Callback):
    def __init__(self, logdir, sample_image):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(logdir)
        self.sample_image = sample_image

    def on_epoch_end(self, epoch, logs={}):
        # Building inference model
        _image = tf.keras.Input(shape=[None, None, 3], name="image")
        _predictions = self.model(_image, training=False)
        _detections = build_decoding_layer(
            2, confidence_threshold=0.5)(_image, _predictions)
        inference_model = tf.keras.Model(inputs=_image, outputs=_detections)

        def prepare_image(image):
            image, _, ratio = resize_and_pad_image(image, jitter=None)
            image = tf.keras.applications.resnet.preprocess_input(image)
            return tf.expand_dims(image, axis=0), ratio

        # Read in the image file.
        image = cv2.imread(self.sample_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img, ratio = prepare_image(image)

        # Generating detections.
        detections = inference_model.predict(img)

        # Parse the result.
        num_detections = detections.valid_detections[0]
        boxes = detections.nmsed_boxes[0][:num_detections] / ratio
        scores = detections.nmsed_scores[0][:num_detections]
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = np.array(box, dtype=int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, "face:{:.2}".format(
                score), (x1, y1-5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

        with self.file_writer.as_default():
            # tf.summary needs a 4D tensor
            img_tensor = tf.expand_dims(image, 0)
            tf.summary.image("test-sample", img_tensor, step=epoch)

        return
