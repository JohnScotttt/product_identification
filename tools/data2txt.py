import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def data2txt(data_root_path, txt_path="total.txt"):
    if os.path.exists(txt_path):
        raise Exception("total.txt is already exist")
    else:
        categories = os.listdir(data_root_path)

        with open(txt_path, "w", encoding="utf-8") as f:
            for category in tqdm(categories):
                category_path = os.path.join(data_root_path, category)
                files = os.listdir(category_path)

                for file in tqdm(files, leave=False):
                    file_path = os.path.join(category_path, file)
                    f.write(file_path + "\t" + str(category) + "\n")

    print("Finished")

def data_split(total_txt, train_txt, val_txt):
    with open(total_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()
    train, val = train_test_split(lines, test_size=0.2, random_state=42)

    with open(train_txt, "w", encoding="utf-8") as f_train:
        f_train.writelines(train)
    with open(val_txt, "w", encoding="utf-8") as f_val:
        f_val.writelines(val)

    print("Finished")