from torch.utils.data import Dataset
from PIL import Image
import cv2

class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.imgs = []
        for line in lines:
            line = line.strip("\n").rstrip().split("\t")[0]
            self.imgs.append(line)
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = cv2.imread(img_path)
        if self.transform is not None:
            img_q = self.transform(img)
            img_k = self.transform(img)
        return img_q, img_k

    def __len__(self):
        return len(self.imgs)