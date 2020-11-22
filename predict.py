import tensorflow as tf
from network import build_model

if __name__ == "__main__":

    # Initializing the model
    model = build_model()

    # Loading weights. Change this to `model_dir` when not using the downloaded
    # weights.
    weights_dir = "data"

    latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
    model.load_weights(latest_checkpoint)

    # Building inference model

    image = tf.keras.Input(shape=[None, None, 3], name="image")
    predictions = model(image, training=False)
    detections = DecodePredictions(
        confidence_threshold=0.5)(image, predictions)
    inference_model = tf.keras.Model(inputs=image, outputs=detections)

    # Generating detections

    def prepare_image(image):
        image, _, ratio = resize_and_pad_image(image, jitter=None)
        image = tf.keras.applications.resnet.preprocess_input(image)
        return tf.expand_dims(image, axis=0), ratio

    val_dataset = tfds.load("coco/2017", split="validation", data_dir="data")
    int2str = dataset_info.features["objects"]["label"].int2str

    for sample in val_dataset.take(2):
        image = tf.cast(sample["image"], dtype=tf.float32)
        input_image, ratio = prepare_image(image)
        detections = inference_model.predict(input_image)
        num_detections = detections.valid_detections[0]
        class_names = [
            int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
        ]
        visualize_detections(
            image,
            detections.nmsed_boxes[0][:num_detections] / ratio,
            class_names,
            detections.nmsed_scores[0][:num_detections],
        )
