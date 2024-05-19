import argparse
import json
import matplotlib.pyplot as plt
from os.path import splitext
from math import ceil

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True)

args = parser.parse_args()

file = args.file

history = json.load(open(file, "r"))

metrics = list(history.keys())[:(len(history.keys()) // 2)]

fig, ax = plt.subplots(figsize=(15, 10), nrows=ceil(len(metrics) / 2), ncols=2)

for i, metric in enumerate(metrics):
    x, y = i // 2, i % 2
    ax[x, y].plot(history[metric])
    ax[x, y].plot(history[f'val_{metric}'])
    ax[x, y].set_title(metric)
    # ax[x, y].set_ylim(0.0, 1.0)
    ax[x, y].legend(['train', 'val'])

if len(metrics) % 2:
    fig.delaxes(ax[len(metrics) // 2, 1])

plt.savefig(f'{splitext(file)[0]}.png', bbox_inches='tight')