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

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Cutout
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, SimpleRGBImageDecoder

import cv2

def display_img(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def convert_dataset(ds):
    if ds=='cifar':
        # cifar
        train_set = datasets.CIFAR10('../datasets/cifar10', train = True, transform = None, target_transform = None, download = False)
        test_set = datasets.CIFAR10('../datasets/cifar10', train = False, transform = None, target_transform = None, download = False)
        train_write_path = 'cifar_train.beton'
        test_write_path = 'cifar_test.beton'
        max_reso=32
    elif ds == 'imagenet':
        train_set = datasets.ImageNet('../datasets/imagenet', split="train", transform = None, target_transform = None)
        test_set = datasets.ImageNet('../datasets/imagenet', split="val", transform = None, target_transform = None)
        train_write_path = 'imagenet_train.beton'
        test_write_path = 'imagenet_test.beton'
        max_reso=256

    train_writer = DatasetWriter(train_write_path, {
        # Tune options to optimize dataset size, throughput at train-time
        'image': RGBImageField(
            max_resolution=max_reso,
            jpeg_quality=10
        ),
        'label': IntField()
    })

    test_writer = DatasetWriter(test_write_path, {
        # Tune options to optimize dataset size, throughput at train-time
        'image': RGBImageField(
            max_resolution=max_reso,
            jpeg_quality=10
        ),
        'label': IntField()
    })

    # Write dataset
    train_writer.from_indexed_dataset(train_set)
    test_writer.from_indexed_dataset(test_set)

def peek_dataset(ds):
    decoder = SimpleRGBImageDecoder()
    image_pipeline = [decoder, ToTensor(), ToTorchImage()]
    label_pipeline = [IntDecoder(), ToTensor()]

    # Pipeline for each data field
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    }

    if ds == 'cifar':
        write_path = '../datasets/cifar10/cifar_test.beton'
    elif ds == 'imagenet':
        write_path = '../datasets/imagenet/imagenet_test.beton'
    # Replaces PyTorch data loader (`torch.utils.data.Dataloader`)
    loader = Loader(write_path, batch_size=1, num_workers=1,
                    order=OrderOption.SEQUENTIAL, pipelines=pipelines)
    for i,(imgs, labels) in enumerate(loader):
        print('labels', labels)
        print('imgs', imgs)
        img = cv2.cvtColor(imgs[0].permute(1,2,0).cpu().numpy(), cv2.COLOR_BGR2RGB)
        display_img(img)



if __name__ == '__main__':
    #convert_dataset('imagenet')
    peek_dataset('cifar')
