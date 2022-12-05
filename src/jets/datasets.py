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
    def __init__(self, data_config="QCD1f", transform=None, QCD_bg=True):
        # Class that manages the data of an experiment and loads it when needed. Modify it when needed
        if data_config == "QCD1f":
            self.bg_dataset = Jetimagedataset(
                transform=transform, path="/mnt/c/WORK/DATA/jet_images/Xtra-100KQCD-pre3-2.pickle")
            self.train_set = self.bg_dataset
            self.val_set = Jetimagedataset(
                transform=transform, path="/mnt/c/WORK/DATA/jet_images/Xval-40KQCD-pre3-2.pickle")
            self.test_bg = Jetimagedataset(
                transform=transform, path="/mnt/c/WORK/DATA/jet_images/Xtes-40KQCD-pre3-2.pickle")
            self.test_sg = Jetimagedataset(
                transform=transform, path="/mnt/c/WORK/DATA/jet_images/Xtes-40Ktop-pre3-2.pickle")

        if data_config == "top1f":
            self.bg_dataset = Jetimagedataset(
                transform=transform, path="/mnt/c/WORK/DATA/jet_images/Xtra-100Ktop-pre3-2.pickle")
            self.train_set = self.bg_dataset
            self.val_set = Jetimagedataset(
                transform=transform, path="/mnt/c/WORK/DATA/jet_images/Xval-40Ktop-pre3-2.pickle")
            self.test_bg = Jetimagedataset(
                transform=transform, path="/mnt/c/WORK/DATA/jet_images/Xtes-40Ktop-pre3-2.pickle")
            self.test_sg = Jetimagedataset(
                transform=transform, path="/mnt/c/WORK/DATA/jet_images/Xtes-40KQCD-pre3-2.pickle")
