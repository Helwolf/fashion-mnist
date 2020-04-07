from torch.utils.data import Dataset
from data_util import get_train_data, get_test_data
from torchvision import transforms
from skimage.util import random_noise
import numpy as np
from torchvision.datasets import MNIST

class RandomNoise(object):
    def __init__(self, mean=2, var=0.1):
        self.mean = mean
        self.var = var

    def __call__(self, img):
        noise = np.random.normal(self.mean, self.var, img.shape)
        return (img+noise).astype(np.uint8)

train_transpose = transforms.Compose([
    transforms.RandomApply([
        RandomNoise()
    ]),
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.RandomApply([
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Compose([
            transforms.Resize((24, 24)),
            transforms.Pad(2)
        ])
    ]),
    transforms.ToTensor(),
])

test_transpose = transforms.Compose([
    transforms.ToTensor(),
])

class FashionMnist(Dataset):

    def __init__(self, data_type='train', data=None):

        self.data_type = data_type
        if data_type == 'train':
            self.transform = test_transpose
            self.data, self.label = data
        else:
            self.transform = test_transpose
            if data_type =='val':
                self.data, self.label = data
            else:
                self.data = data

    def __getitem__(self, index):

        if self.data_type == 'train' or self.data_type == 'val':
            return self.transform(self.data[index]), self.label[index]
        else:
            return self.transform(self.data[index])

    def __len__(self):
        return self.data.shape[0]

