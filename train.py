"""Training code for RetinaNet face detection."""

import os
import tensorflow as tf

from dataset import load_datasets
from losses import RetinaNetLoss
from network import build_model

if __name__ == "__main__":

    # Setting up training parameters
    model_dir = "retinanet/"

    num_classes = 2
    batch_size = 32

    learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
    learning_rate_boundaries = [125, 250, 500, 240000, 360000]
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=learning_rate_boundaries, values=learning_rates)

    loss_fn = RetinaNetLoss(num_classes)
    optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)

    # Initializing and compiling model
    model = build_model(num_classes)
    model.compile(loss=loss_fn, optimizer=optimizer)

    # Setting up callbacks
    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
            monitor="loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,),
        tf.keras.callbacks.TensorBoard()
    ]

    # Load the WIDER dataset using TensorFlow Datasets
    train_dataset, val_dataset = load_datasets(batch_size=1)

    # Training the model
    epochs = 20

    # Running 100 training and 50 validation steps,
    model.fit(train_dataset,
              validation_data=val_dataset.take(50),
              epochs=epochs,
              callbacks=callbacks_list,
              verbose=1)
