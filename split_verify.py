import argparse
import pandas as pd
from utils import load_dataset
from os.path import normpath, basename, splitext

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)

args = parser.parse_args()

name = args.name

train, val, test, labels = load_dataset(name)

trainid = train['id'].apply(lambda x: int(splitext(basename(normpath(x)))[0])).to_list()
valid = val['id'].apply(lambda x: int(splitext(basename(normpath(x)))[0])).to_list()
testid = test['id'].apply(lambda x: int(splitext(basename(normpath(x)))[0])).to_list()

print('train size:', len(trainid))
print('val size:', len(valid))
print('test size:', len(testid))

print('train + val dupes:', len(set(trainid) & set(valid)))
print('train + test dupes:', len(set(trainid) & set(testid)))
print('val + test dupes:', len(set(valid) & set(testid)))
