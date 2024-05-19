import tensorflow as tf
from tensorflow.keras import layers  # type: ignore
import pandas as pd
import numpy as np
import os

from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import train_test_split

BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 512
AUTOTUNE = tf.data.experimental.AUTOTUNE

@tf.function
def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost) # average on all labels
    return macro_cost

@tf.function
def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)
    
    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive
        
    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1

def parse_function(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [256, 256])
    return img, label
def create_dataset(filenames, labels):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)
    #dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
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
            #layers.Dropout(0.05),
            layers.Conv2D(32, 5, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            #layers.Dropout(0.1),
            layers.Conv2D(64, 5, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            #layers.Dense(256, activation="relu"),
            
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(number_of_classes, name="outputs", activation="sigmoid"),
        ]
    )

    model.build(input_shape=(None, 256, 256, 3))

    adam = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(
        optimizer=adam,
        loss=macro_soft_f1,
        metrics=[
            "accuracy",
            "mae",
            macro_f1
            #tf.keras.metrics.F1Score(threshold=0.5, average="macro"),
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
