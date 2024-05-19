import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--model", type=str, required=False)
parser.add_argument("--plot", action="store_true")

args = parser.parse_args()

if args.model and not os.path.exists(f"{args.name}/model/{args.model}"):
    raise FileNotFoundError(f"{args.name}/model/{args.model} not found")

import json
import matplotlib.pyplot as plt
from utils import create_model, create_dataset, load_dataset


name = args.name

history = json.load(open(f"{name}/history.json", "r"))

fig, ax = plt.subplots(figsize=(20, 10), nrows=2, ncols=2)
fig.suptitle(f"{name} metrics")

print(list(history.keys()))

ax[0, 0].plot(history["accuracy"])
ax[0, 0].plot(history["val_accuracy"])
ax[0, 0].set_title("Accuracy")
ax[0, 0].legend(["train", "val"])

ax[0, 1].plot(history["loss"])
ax[0, 1].plot(history["val_loss"])
ax[0, 1].set_title("Loss")
ax[0, 1].legend(["train", "val"])

ax[1, 0].plot(history["macro_f1"])
ax[1, 0].plot(history["val_macro_f1"])
ax[1, 0].set_title("F1 score")
ax[1, 0].legend(["train", "val"])

ax[1, 1].plot(history["mae"])
ax[1, 1].plot(history["val_mae"])
ax[1, 1].set_title("Mean absolute error")
ax[1, 1].legend(["train", "val"])

plt.savefig(f"{name}/metrics.png")

if args.plot:
    exit()

train, val, test, labels = load_dataset(name)

x_train, y_train = train['id'], train['labels']
x_val, y_val = val['id'], val['labels']
x_test, y_test = test['id'], test['labels']

train_dataset = create_dataset(x_train, y_train)
val_dataset = create_dataset(x_val, y_val)
test_dataset = create_dataset(x_test, y_test)

if args.model:
    model = create_model(len(labels))
    model.load_weights(f"{name}/model/{args.model}")

    model.evaluate(val_dataset)
else:
    results = []

    for model_name in sorted(
        os.listdir(f"{name}/model"), key=lambda x: int(x.split("-")[0])
    )[-50:]:
        print(model_name)
        model = create_model(len(labels))
        model.load_weights(f"{name}/model/{model_name}")

        loss, accuracy, mae, f1 = model.evaluate(val_dataset)
        results.append((model_name, loss, accuracy, mae, f1))

    results = sorted(results, key=lambda x: x[-1], reverse=True)
    # save results as json
    with open(f"{name}/results.json", "w") as f:
        json.dump(results, f)
