from torch.utils.data import Dataset
import numpy as np
import random
from PIL import Image 
import os


# mock images (get a lot of this) and text encodings from large T5
class read_dataset(Dataset):
    def __init__(self, path_npy, path_img):
        self.dataset = np.load(path_npy, allow_pickle=True)
        self.path_img = path_img

    def __getitem__(self, index):
        id = self.dataset[index][0]
        text = self.dataset[index][1].astype(np.int64)
        if self.path_img == None:
            with Image.open('fake_img.png') as image:
                if random.random() < 0.5:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                image = np.array(image).astype(np.float32) / 255.0
        else:
            with Image.open(self.path_img + '/' + id + '.png') as image:
                if random.random() < 0.5:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        return id, text, image

    def __len__(self):
        return len(self.dataset)

        