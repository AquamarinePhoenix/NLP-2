import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from nlplib.config import WORDS, SEED
from nlplib.write import write_file

def run_CNN(
    train_dimAudio: pd.DataFrame,
    test_dimAudio: pd.DataFrame,
    *,
    epochs: int = 15,
    batch_size: int = 32,
):
    tf.keras.utils.set_random_seed(SEED)

    label_to_id = {word: i for i, word in enumerate(WORDS)}

    X_train = np.stack(train_dimAudio["features"].to_numpy())
    X_test = np.stack(test_dimAudio["features"].to_numpy())

    y_train = train_dimAudio["word"].map(label_to_id).to_numpy()
    y_test = test_dimAudio["word"].map(label_to_id).to_numpy()

    X_train = X_train[..., np.newaxis].astype("float32")
    X_test = X_test[..., np.newaxis].astype("float32")

    mean = X_train.mean()
    std = X_train.std() + 1e-8

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    model = keras.Sequential([
        layers.Input(shape=(40, 101, 1)),

        layers.Conv2D(
            16, kernel_size=(3, 3),
            padding="same", activation="relu"
        ),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(
            32, kernel_size=(3, 3),
            padding="same", activation="relu"
        ),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose = 0)

    write_file(f"Test loss: {test_loss:.4f}")
    write_file(f"Test accuracy: {test_accuracy:.4f}")
    return model, mean, std, history