import tensorflow as tf
from tensorflow.keras import layers  # type: ignore
import pandas as pd
import numpy as np
import os

from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import train_test_split

BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE


def parse_function(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [256, 256])
    return img, label


def create_dataset(filenames, labels):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset


def create_model(number_of_classes):

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.3),
            layers.RandomContrast(0.5),
        ]
    )

    model = tf.keras.Sequential(
        [
            data_augmentation,
            layers.Conv2D(16, 5, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.1),
            layers.Conv2D(32, 5, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Conv2D(64, 5, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(64, activation="relu"),
            layers.Dense(number_of_classes, name="outputs", activation="sigmoid"),
        ]
    )

    model.build(input_shape=(None, 256, 256, 3))

    adam = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(
        optimizer=adam,
        loss=tf.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            "mae",
            tf.keras.metrics.F1Score(threshold=0.5, average="macro"),
        ],
    )

    return model


def split_data(x, y, ratio, random_state=None, merge_train_val=False):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=ratio, random_state=random_state
    )
    x_test, x_val, y_test, y_val = train_test_split(
        x_test, y_test, test_size=0.5, random_state=random_state
    )

    if merge_train_val:
        x_train = np.vstack((x_train, x_val))
        y_train = np.vstack((y_train, y_val))

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def iterative_split_data(x, y, ratio, random_state=None, merge_train_val=False):
    x_train, y_train, x_test, y_test = iterative_train_test_split(
        x,
        y,
        test_size=ratio,
    )

    x_test, y_test, x_val, y_val = iterative_train_test_split(
        x_test, y_test, test_size=0.5
    )

    if merge_train_val:
        x_train = np.vstack((x_train, x_val))
        y_train = np.vstack((y_train, y_val))

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def load_dataset(name):
    train = pd.read_json(f"{name}/train.json", orient="records")
    val = pd.read_json(f"{name}/val.json", orient="records")
    test = pd.read_json(f"{name}/test.json", orient="records")
    labels = pd.read_json(f"{name}/labels.json", orient="records")

    return train, val, test, labels


def get_id(x):
    return int(x.split(".")[0])


def split_and_save(data, labels, ratio, name, random_state=None, merge_train_val=False, iterative=False):
    if iterative:
        train, val, test = iterative_split_data(
            data, labels, ratio, random_state=random_state, merge_train_val=merge_train_val
        )
    else:
        train, val, test = split_data(
            data, labels, ratio, random_state=random_state, merge_train_val=merge_train_val
        )
    

    train = pd.DataFrame(zip(*train), columns=["id", "labels"])
    val = pd.DataFrame(zip(*val), columns=["id", "labels"])
    test = pd.DataFrame(zip(*test), columns=["id", "labels"])

    train['id'] = train['id'].apply(lambda x: x[0])
    val['id'] = val['id'].apply(lambda x: x[0])
    test['id'] = test['id'].apply(lambda x: x[0])

    save(train, val, test, name)


def save(train, val, test, name):
    if not os.path.exists(name):
        os.makedirs(name)

    train.to_json(f"{name}/train.json", orient="records")
    val.to_json(f"{name}/val.json", orient="records")
    test.to_json(f"{name}/test.json", orient="records")
