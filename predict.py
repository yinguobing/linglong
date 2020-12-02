from argparse import ArgumentParser

import cv2
import numpy as np
import tensorflow as tf

from network import build_decoding_layer, build_model
from preprocessing import resize_and_pad_image

# Take arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
parser.add_argument("--output", type=str, default=None,
                    help="Save the processed video file.")
args = parser.parse_args()


if __name__ == "__main__":

    # Initializing the model
    model = build_model(2)

    # Loading weights.
    latest_checkpoint = tf.train.latest_checkpoint("checkpoints")
    model.load_weights(latest_checkpoint)

    # Building inference model
    image = tf.keras.Input(shape=[None, None, 3], name="image")
    predictions = model(image, training=False)
    detections = build_decoding_layer(
        2, confidence_threshold=0.6)(image, predictions)
    inference_model = tf.keras.Model(inputs=image, outputs=detections)

    # Generating detections
    def prepare_image(image):
        image, _, ratio = resize_and_pad_image(image, jitter=None)
        image = tf.keras.applications.resnet.preprocess_input(image)
        return tf.expand_dims(image, axis=0), ratio

    # Video source from webcam or video file.
    video_src = args.cam if args.cam is not None else args.video
    if video_src is None:
        print("Warning: video source not assigned, default webcam will be used.")
        video_src = 0

    cap = cv2.VideoCapture(video_src)
    if video_src == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    # Save the processed video file.
    if args.output:
        fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        video_writer = cv2.VideoWriter(
            args.output, fourcc, frame_rate, (width, height))

    while True:
        # Read frame, crop it, flip it, suits your needs.
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        # If frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Read in and preprocess the sample image
        input_image, ratio = prepare_image(frame)

        # Do prediction.
        detections = inference_model.predict(input_image)

        # Parse the detection result.
        num_detections = detections.valid_detections[0]
        boxes = detections.nmsed_boxes[0][:num_detections] / ratio
        scores = detections.nmsed_scores[0][:num_detections]
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = np.array(box, dtype=int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "face:{:.2}".format(
                score), (x1, y1-5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

        # Save the output in video file.
        if args.output:
            video_writer.write(frame)

        # Show the result in windows.
        cv2.imshow('image', frame)
        if cv2.waitKey(1) == 27:
            break
