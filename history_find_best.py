import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True)

args = parser.parse_args()

file = args.file

history = json.load(open(file, "r"))

macro, epoch = max([(j, i) for i, j in enumerate(history['val_macro_f1'])])
print('epoch:', epoch + 1, 'val_macro_f1', macro)