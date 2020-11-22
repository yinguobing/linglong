import tensorflow.keras as keras
import cv2


class LogImages(keras.callbacks.Callback):
    def __init__(self, logdir, sample_image):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(logdir)
        self.sample_image = sample_image

    def on_epoch_end(self, epoch, logs={}):
        # Read in the image file.
        image = cv2.imread(self.sample_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(image, (128, 128))
        img = normalize(img)

        prediction = self.model.predict(tf.expand_dims(img, 0))[0]
        faces = decode(prediction, 0.1)
        draw_face_boxes(image, faces)

        with self.file_writer.as_default():
            # tf.summary needs a 4D tensor
            img_tensor = tf.expand_dims(image, 0)
            tf.summary.image("test-sample", img_tensor, step=epoch)

        return
