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
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, SimpleRGBImageDecoder, CenterCropRGBImageDecoder
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.compiler import Compiler
from dataclasses import replace
from PIL import Image

import cv2

class ImagenetTransform(Operation):

    # Return the code to run this operation
    def generate_code(self):
        parallel_range = Compiler.get_iterator()
        def imagenet_trans(images, dst):
            '''
            images - b x h x w x c
            dst - b x 224 x 224 x c
            '''
            from timm.data import create_transform
            transform = create_transform(
                input_size=224,
                is_training=True,
                color_jitter=0.4,
                auto_augment='rand-m9-mstd0.5-inc1',
                re_prob=0.25,
                re_mode='pixel',
                re_count=1,
                interpolation='bicubic',
            )
            transform.transforms[0] = transforms.ToPILImage()
            del transform.transforms[-2] # no normalization
            for i in parallel_range(images.shape[0]):
                img_trans = transform(images[i])
                dst[i] = img_trans
            return dst
        imagenet_trans.is_parallel = True
        return imagenet_trans

    def declare_state_and_memory(self, previous_state):
        h, w, c = previous_state.shape
        new_shape = (c, 224, 224)
        new_dtype = torch.float32 
 
        # Everything in the state stays the same other than the shape
        # States are immutable, so we have to edit them using the
        # dataclasses.replace function
        new_state = replace(previous_state, shape=new_shape, dtype=new_dtype, jit_mode=False)
 
        # We need to allocate memory for the new images
        # so below, we ask for a memory allocation whose width and height is
        # half the original image, with the same type
        # (shape=(,)) of the same type as the image data
        mem_allocation = AllocationQuery(new_shape, new_dtype)
        return (new_state, mem_allocation)

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
    if ds == 'cifar':
        write_path = '../datasets/cifar10/cifar_test.beton'
        decoder = SimpleRGBImageDecoder()
        image_pipeline = [decoder, ToTensor(), ToTorchImage()]
    elif ds == 'imagenet':
        write_path = '../datasets/imagenet/imagenet_test.beton'
        decoder = RandomResizedCropRGBImageDecoder((224,224))
        image_pipeline = [decoder, ImagenetTransform()]
        '''
        decoder = CenterCropRGBImageDecoder((224,224),224/256)
        image_pipeline = [decoder, ToTensor(), ToTorchImage()]
        '''
    label_pipeline = [IntDecoder(), ToTensor()]

    # Pipeline for each data field
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    }

    # Replaces PyTorch data loader (`torch.utils.data.Dataloader`)
    loader = Loader(write_path, batch_size=1, num_workers=1,
                    order=OrderOption.SEQUENTIAL, pipelines=pipelines)
    for i,(imgs, labels) in enumerate(loader):
        print('labels', labels)
        print('imgs', imgs)
        print('min max',imgs.min(),imgs.max())
        print(imgs.shape)
        img = cv2.cvtColor(imgs[0].permute(1,2,0).cpu().numpy(), cv2.COLOR_BGR2RGB)
        display_img(img)



if __name__ == '__main__':
    #convert_dataset('imagenet')
    peek_dataset('imagenet')
