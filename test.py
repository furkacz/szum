import argparse
import os
import json
from utils import create_model, create_dataset, load_dataset

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

results = { 'train' : {}, 'val' : {}, 'test' : {} }
results['train']['loss'], results['train']['accuracy'], results['train']['mae'], results['train']['mse'], results['train']['macro_f1'] = model.evaluate(train_dataset)
results['val']['loss'], results['val']['accuracy'], results['val']['mae'], results['val']['mse'], results['val']['macro_f1'] = model.evaluate(val_dataset)
results['test']['loss'], results['test']['accuracy'], results['test']['mae'], results['test']['mse'], results['test']['macro_f1'] = model.evaluate(test_dataset)

with open(f"{name}/results.json", "w") as f:
    json.dump(results, f)
