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

train_count = [reduce(lambda a, b: [x + y for x, y in zip(a, b)], train['labels'])]
val_count = [reduce(lambda a, b: [x + y for x, y in zip(a, b)], val['labels'])]
test_count = [reduce(lambda a, b: [x + y for x, y in zip(a, b)], test['labels'])]
# total_count = [[a + b + c for a, b, c in zip(train_count[0], val_count[0], test_count[0])]]

norm_train_count = normalize(train_count)
norm_val_count = normalize(val_count)
norm_test_count = normalize(test_count)
# norm_total_count = normalize(total_count)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
# ax1.bar(range(len(norm_total_count[0])), norm_total_count[0])
ax1.bar(range(len(train_count[0])), train_count[0])
ax1.set_title('train')
ax2.bar(range(len(val_count[0])), val_count[0])
ax2.set_title('val')
ax3.bar(range(len(test_count[0])), test_count[0])
ax3.set_title('test')
plt.savefig(f'{name}.png', bbox_inches='tight')
