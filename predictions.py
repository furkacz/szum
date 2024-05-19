import argparse
import os
import numpy as np
import json
from utils import create_model, create_dataset, load_dataset
from sklearn.preprocessing import MultiLabelBinarizer

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--model", type=str, required=False)

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
model.load_weights(args.model)

with open(f"{name}/labels.json", "r") as f:
    labels = json.load(f)

mlb = MultiLabelBinarizer()
mlb.fit_transform([labels])
original = mlb.inverse_transform(np.array(y_test))

results = np.round(model.predict(test_dataset))
predicted = mlb.inverse_transform(np.array(results))

with open(name + "/original.json", "w") as f:
    json.dump(original, f)

with open(name + "/predicted.json", "w") as f:
    json.dump(predicted, f)
