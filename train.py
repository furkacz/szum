import tensorflow as tf
from utils import create_model, create_dataset, load_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--epochs", type=int, required=True)

args = parser.parse_args()

name = args.name

train, val, test, labels = load_dataset(name)

x_train, y_train = train['id'].to_list(), train['labels'].to_list()
x_val, y_val = val['id'].to_list(), val['labels'].to_list()
x_test, y_test = test['id'].to_list(), test['labels'].to_list()

train_dataset = create_dataset(x_train, y_train)
val_dataset = create_dataset(x_val, y_val)
test_dataset = create_dataset(x_test, y_test)

model = create_model(len(labels))

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=name + "/model/{epoch:02d}-{val_loss:.2f}.keras",
    monitor="val_macro_f1",
    verbose=1,
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_macro_f1", patience=5, verbose=1, start_from_epoch=0
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=args.epochs,
    callbacks=[model_checkpoint],
)

model.evaluate(test_dataset)

# save history
import json

with open(name + "/history.json", "w") as f:
    json.dump(history.history, f)
