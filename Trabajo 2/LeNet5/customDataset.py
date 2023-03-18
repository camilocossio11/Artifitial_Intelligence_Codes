import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import PIL
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


class NumbersDataset(Dataset):

    def __init__(self, root, image_dir, csv_file, transform=None):
        self.root = root
        self.image_dir = image_dir
        self.image_files = pd.read_csv(
            csv_file, sep=';', header=None).iloc[:, 0]
        self.data = pd.read_csv(csv_file, sep=';', header=None).iloc[:, 1]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_name = os.path.join(self.image_dir, self.image_files[index])
        image = PIL.Image.open(image_name)
        label = self.data[index]
        if self.transform:
            image = self.transform(image)
        return (image, label)
