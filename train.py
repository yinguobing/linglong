"""Training code for RetinaNet face detection."""

import os
from argparse import ArgumentParser

import tensorflow as tf

from dataset import load_datasets
from losses import RetinaNetLoss
from network import build_model
from callbacks import LogImages

parser = ArgumentParser()
parser.add_argument("--epochs", default=60, type=int,
                    help="Number of training epochs.")
parser.add_argument("--initial_epoch", default=0, type=int,
                    help="From which epochs to resume training.")
parser.add_argument("--batch_size", default=2, type=int,
                    help="Training batch size.")
parser.add_argument("--export_only", default=False, type=bool,
                    help="Save the model without training.")
parser.add_argument("--eval_only", default=False, type=bool,
                    help="Evaluate the model without training.")
args = parser.parse_args()


if __name__ == "__main__":
    # Set the TensorFlow training and validation record files' path.
    record_train = "/home/robin/data/face/wider/tfrecord/wider_train.record"
    record_val = "/home/robin/data/face/wider/tfrecord/wider_val.record"

    # Checkpoint is used to resume training.
    checkpoint_dir = "./checkpoints/"

    # Save the model for inference later.
    export_dir = "./exported"

    # Log directory will keep training logs like loss/accuracy curves.
    log_dir = "./logs"

    # A sample image used to log the model's behavior during training.
    log_image = "./docs/family.jpg"

    # Setup the model
    num_classes = 2
    model = build_model(num_classes)

    # Model built. Restore the latest model if checkpoints are available.
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print("Checkpoint directory created: {}".format(checkpoint_dir))

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Checkpoint found: {}, restoring...".format(latest_checkpoint))
        model.load_weights(latest_checkpoint)
        print("Checkpoint restored: {}".format(latest_checkpoint))
    else:
        print("Checkpoint not found. Model weights will be initialized randomly.")

    # If the restored model is ready for inference, save it and quit training.
    if args.export_only:
        if latest_checkpoint is None:
            print("Warning: Model not restored from any checkpoint.")
        print("Saving model to {} ...".format(export_dir))
        model.save(export_dir)
        print("Model saved at: {}".format(export_dir))
        quit()

    # Compile the model for training.
    learning_rate_schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[125, 250, 500, 240000, 360000],
        values=[2.5e-07, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05])
    optimizer = tf.optimizers.SGD(learning_rate=learning_rate_schedule)
    model.compile(loss=RetinaNetLoss(num_classes), optimizer=optimizer)

    # Setting up callbacks
    callbacks_list = [tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "linglong"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        LogImages(log_dir, log_image)]

    # Load the WIDER dataset using TensorFlow Datasets
    dataset_train = load_datasets(record_train, args.batch_size, True)
    dataset_val = load_datasets(record_val, args.batch_size, False)

    # Training
    model.fit(dataset_train,
              validation_data=dataset_val.take(50),
              epochs=args.epochs,
              callbacks=callbacks_list,
              initial_epoch=args.initial_epoch,
              verbose=1)
