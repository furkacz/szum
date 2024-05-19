import numpy as np
import pandas as pd
import os

from utils import get_id, save
from collections import Counter, namedtuple


PATH = "fashion-dataset"
RANDOM_AND_ITERATIVE=False
RANDOM_SEED=2137

data = pd.read_csv(f"{PATH}/styles.csv", on_bad_lines="skip") # wczytanie metadanych do każdego obrazka

data.loc[data["articleType"].str.contains("Shoes"), "articleType"] = "Shoes" # mergujemy wszystkie labele zawierające shoes w jeden
data.loc[data["gender"] == "Boys", "gender"] = "Men" # mergujemy boys i men jako men
data.loc[data["gender"] == "Girls", "gender"] = "Women" # mergujemy girls i women jako women

directory = list(map(get_id, os.listdir(f"{PATH}/images"))) # wczytujemy ID obrazków
top10_colors = data["baseColour"].value_counts().head(10).index.to_list() # znajduje które 10 kolorów się najczęściej powtarza
genders = data["gender"].value_counts().index.to_list() # zwraca 3 labele: men women unisex
seasons = data["season"].value_counts().index.to_list() # zwraca 4 labele: winter summer fall spring
usages = data["usage"].value_counts() # pobieramy wszystkie możliwe usage
usages = usages[usages > 1000].index.to_list() # zapisujemy tylko labele usage ktore wystapily co najmniej 1000 razy

# zbierz wszystkie rekordy ktore spelniaja wszystkie warunki
data = data[data["id"].isin(directory)]
data = data[data["baseColour"].isin(top10_colors)]
data = data[data["gender"].isin(genders)]
data = data[data["usage"].isin(usages)]
data = data[data["season"].isin(seasons)]

# ustalamy fallback kategorie
TYPES = ["masterCategory", "subCategory", "articleType"]

master_sub_type_table = data[TYPES].value_counts() # wszystkie mozliwe wartosci dla master category
master_sub_table = data[TYPES[0:2]].value_counts() # wszystkie mozliwe wartosci dla subcategory
master_table = data[TYPES[0:1]].value_counts() # wszystkie mozliwe wartosci articletype

# -- START -- patrzymy dla kazdej kategorii czy jest minimum LIMIT wartosci
# jak nie to mergujemy do subcategory
# jak subcategory jest za male to mergujemy do category
LIMIT = 1500

Configuration = namedtuple("Configuration", TYPES, defaults=[None, None, None])

counter = Counter()

for categories, count in master_sub_type_table.items():
    master, sub, article = categories

    if count > LIMIT:
        counter[Configuration(master, sub, article)] += count
    elif master_sub_table[(master, sub)] > LIMIT:
        counter[Configuration(master, sub)] += count
    elif master_table[master] > LIMIT:
        counter[Configuration(master)] += count

values = set().union(*counter)

if None in values:
    values.remove(None)

# na koniec mamy wszystkie kategorie w values
# --END--

# zbiera wszystkie labele ktore spelniaja to values
def gatherer(values):
    def func(row):
        labels = set()

        for key in reversed(TYPES):
            if row[key] in values:
                labels.add(row[key])
                break

        for key in ["baseColour", "gender", "season", "usage"]:
            labels.add(row[key])

        return list(labels)

    return func


data = data[
    data["masterCategory"].isin(values)
    | data["subCategory"].isin(values)
    | data["articleType"].isin(values)
] # tutaj odfilrtujemy bardzo bardzo male liczne zbiory dla danego label tak male ze master nawet nie przekroczyl LIMIT

gather_labels = gatherer(values)
files = data["id"].apply(lambda x: os.path.join(f"{PATH}/images", str(x) + ".jpg"))
labels = data.apply(gather_labels, axis=1)

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
xfiles = files.to_numpy()[..., np.newaxis]
ylabels = mlb.fit_transform(labels).astype('float32')

def random_split_data(data, labels, ratio, random_state=None):
    samples = pd.DataFrame(zip(data, labels), columns=['id', 'labels'])

    valtest = samples.sample(frac=ratio, replace=False, random_state=random_state)
    train = samples.iloc[samples.index.difference(valtest.index)]

    valtest = valtest.reset_index()

    val = valtest.sample(frac=0.5, replace=False, random_state=random_state)
    test = valtest.iloc[valtest.index.difference(val.index)]

    return train, val, test

train, val, test = random_split_data(xfiles, ylabels, ratio=0.2, random_state=2137)

train['id'] = train['id'].apply(lambda x: x[0])
val['id'] = val['id'].apply(lambda x: x[0])
test['id'] = test['id'].apply(lambda x: x[0])

save(train, val, test, 'split2-random')

pd.DataFrame(mlb.classes_)[0].to_json('split2-random/labels.json', orient='values')