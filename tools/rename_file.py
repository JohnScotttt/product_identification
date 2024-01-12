import os
import shutil
from tqdm import tqdm
import json

def rename_file(data_root_path, new_data_root_path, index_path="index.json"):
    categories = os.listdir(data_root_path)
    index = json.load(open(index_path, "r"))
    inversed_index = {v: k for k, v in index.items()}

    if not os.path.exists(new_data_root_path):
        os.makedirs(new_data_root_path)
    elif os.listdir(new_data_root_path) != []:
        raise Exception("new_data_root_path is not empty")

    for category in tqdm(categories):
        if category not in inversed_index:
            raise Exception(f"{category} is not in index.json")
        
        i = inversed_index[category]
        category_path = os.path.join(data_root_path, category)
        files = os.listdir(category_path)
        new_category_path = os.path.join(new_data_root_path, i)

        if not os.path.exists(new_category_path):
            os.mkdir(new_category_path)

        for j, file in enumerate(tqdm(files, leave=False)):
            extension = '.' + file.split('.')[-1]
            file_path = os.path.join(category_path, file)
            new_file_path = os.path.join(new_category_path, i + '_' + str(j) + extension)
            shutil.copyfile(file_path, new_file_path)

    print("Finished")