import os
import sys
import time
import copy
import math
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField

# cifar
train_set = datasets.CIFAR10('../datasets/cifar10', train = True, transform = None, target_transform = None, download = False)
test_set = datasets.CIFAR10('../datasets/cifar10', train = False, transform = None, target_transform = None, download = False)
train_write_path = 'cifar_train.beton'
test_write_path = 'cifar_test.beton'

train_writer = DatasetWriter(train_write_path, {
    # Tune options to optimize dataset size, throughput at train-time
    'image': RGBImageField(
        max_resolution=32,
        jpeg_quality=10
    ),
    'label': IntField()
})

test_writer = DatasetWriter(test_write_path, {
    # Tune options to optimize dataset size, throughput at train-time
    'image': RGBImageField(
        max_resolution=32,
        jpeg_quality=10
    ),
    'label': IntField()
})

# Write dataset
train_writer.from_indexed_dataset(train_set)
test_writer.from_indexed_dataset(test_set)
