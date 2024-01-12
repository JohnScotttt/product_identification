import os
from tqdm import tqdm
import json

def make_index(data_root_path, index_path="index.json"):
    if os.path.exists(index_path):
        raise Exception("index.json is already exist")
    else:
        categories = os.listdir(data_root_path)
        index = {}

        with open(index_path, 'w', encoding="utf-8") as f:
            for i, category in enumerate(tqdm(categories)):
                index[i] = category
            json.dump(index, f)

    print("Finished")