import pandas as pd
import numpy as np
import os

from utils import get_id, split_and_save


def gather_labels(row):
    labels = set()

    for key in ["articleType", "baseColour", "gender", "season", "usage"]:
        if pd.notna(row[key]):
            labels.add(row[key])

    return list(labels)


PATH = "fashion-dataset"
RANDOM_AND_ITERATIVE=False
RANDOM_SEED=2137

data = pd.read_csv(f"{PATH}/styles.csv", on_bad_lines="skip")


directory = list(map(get_id, os.listdir(f"{PATH}/images")))
data = data[data["id"].isin(directory)]

files = data["id"].apply(lambda x: os.path.join(f"{PATH}/images", str(x) + ".jpg"))
labels = data.apply(gather_labels, axis=1)

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
xfiles = files.to_numpy()[..., np.newaxis]
ylabels = mlb.fit_transform(labels).astype('float32')

if RANDOM_AND_ITERATIVE:
    split_and_save(xfiles, ylabels, 0.2, "split1-random", random_state=RANDOM_SEED, iterative=False)
    pd.DataFrame(mlb.classes_)[0].to_json('split1-random/labels.json', orient='values')

    split_and_save(xfiles, ylabels, 0.2, "split1-iterative", random_state=RANDOM_SEED, iterative=True)
    pd.DataFrame(mlb.classes_)[0].to_json('split1-iterative/labels.json', orient='values')
else:
    split_and_save(xfiles, ylabels, 0.2, "split1", random_state=RANDOM_SEED)
    pd.DataFrame(mlb.classes_)[0].to_json('split1/labels.json', orient='values')


