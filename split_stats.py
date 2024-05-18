import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from functools import reduce
from utils import load_dataset
from sklearn.preprocessing import normalize

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)

args = parser.parse_args()

name = args.name

train, val, test, labels = load_dataset(name)

train_count = normalize([reduce(lambda a, b: [x + y for x, y in zip(a, b)], train['labels'])])
val_count = normalize([reduce(lambda a, b: [x + y for x, y in zip(a, b)], val['labels'])])
test_count = normalize([reduce(lambda a, b: [x + y for x, y in zip(a, b)], test['labels'])])

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.bar(range(len(train_count[0])), train_count[0])
ax2.bar(range(len(val_count[0])), val_count[0])
ax3.bar(range(len(test_count[0])), test_count[0])
plt.savefig(f'{name}.png')
