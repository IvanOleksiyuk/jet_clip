import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
import numpy as np
import pickle
from torchvision import transforms


class Jetimagedataset(Dataset):
    def __init__(self, path="../../BW/image_data_sets/Xtra-100KQCD-pre3-2.pickle", transform=None, target_transform=None, label=0):
        self.images = pickle.load(open(path, "rb"))
        self.labels = np.ones(len(self.images))*label
        self.transform = transform
        self.target_transform = target_transform
        self.batch = 64
        self.in_batch = 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #print("getitem", self.noise)
        image = self.images[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

# If you want to use your own dataset just create one of data settings and put it into the data manager


class DataManager():
    def __init__(self, data_config="QCD1", transform=None, QCD_bg=True):
        # Class that manages the data of an experiment and loads it when needed. Modify it when needed
        if data_config == "QCD1":
            self.bg_dataset = Jetimagedataset(
                transform=transform, path="/mnt/c/WORK/DATA/jet_images/Xtra-100KQCD-pre3-2.pickle")
            self.train_set, self.val_set, self.test_bg = torch.utils.data.random_split(
                self.bg_dataset, [80000, 10000, 10000])
            self.test_sg = Jetimagedataset(
                transform=transform, path="/mnt/c/WORK/DATA/jet_images/Xtra-100Ktop-pre3-2.pickle")
