import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

from collections import namedtuple

import torch
import torchvision

class DatasetBuilder(object):
    # tuple for dataset config
    DC = namedtuple('DatasetConfig', ['mean', 'std', 'input_size', 'num_classes'])
    
    DATASET_CONFIG = {
        'svhn' :   DC([0.43768210, 0.44376970, 0.47280442], [0.19803012, 0.20101562, 0.19703614], 32, 10),
        'cifar10': DC([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784], 32, 10),
    } 

    def __init__(self, name:str, root_path:str):
        """
        Args
        - name: name of dataset
        - root_path: root path to datasets
        """
        if name not in self.DATASET_CONFIG.keys():
            raise ValueError('name of dataset is invalid')
        self.name = name
        self.root_path = os.path.join(root_path, self.name)

    def __call__(self, train:bool, normalize:bool):
        input_size = self.DATASET_CONFIG[self.name].input_size
        transform = self._get_trainsform(self.name, input_size, train, normalize)
        if self.name == 'svhn':
            dataset = torchvision.datasets.SVHN(root=self.root_path, split='train' if self.train else 'test', transform=transform, download=True)
        elif self.name == 'cifar10':
            dataset = torchvision.datasets.CIFAR10(root=self.root_path, train=train, transform=transform, download=True)
        else: 
            raise NotImplementedError 

        return dataset

    def _get_trainsform(self, name:str, input_size:int, train:bool, normalize:bool):
        transform = []

        # arugmentation
        if train:
            transform.extend([
                torchvision.transforms.RandomHorizontalFlip(),
            ])

        else:
            pass

        # to tensor
        transform.extend([torchvision.transforms.ToTensor(),])

        # normalize
        if normalize:
            transform.extend([
                torchvision.transforms.Normalize(mean=self.DATASET_CONFIG[name].mean, std=self.DATASET_CONFIG[name].std),
            ])

        return torchvision.transforms.Compose(transform)
    
    @property
    def input_size(self):
        return self.DATASET_CONFIG[self.name].input_size

    @property
    def num_classes(self):
        return self.DATASET_CONFIG[self.name].num_classes