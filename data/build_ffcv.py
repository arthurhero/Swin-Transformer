# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform

from torch.utils.data import Dataset, DataLoader

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Cutout, Squeeze, NormalizeImage
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, SimpleRGBImageDecoder, CenterCropRGBImageDecoder
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.compiler import Compiler
from dataclasses import replace
from PIL import Image

import cv2

from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler

config = None

class ImagenetTransform(Operation):
    def generate_code(self):
        parallel_range = Compiler.get_iterator()
        def imagenet_trans(images, dst):
            ''' 
            images - b x h x w x c
            dst - b x 224 x 224 x c
            '''
            from timm.data import create_transform
            transform = create_transform(
                input_size=config.DATA.IMG_SIZE,
                is_training=True,
                color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
                auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
                re_prob=config.AUG.REPROB,
                re_mode=config.AUG.REMODE,
                re_count=config.AUG.RECOUNT,
                interpolation=config.DATA.INTERPOLATION,
            )   
            transform.transforms[0] = transforms.ToPILImage()
            for i in parallel_range(images.shape[0]):
                img_trans = transform(images[i])
                img_trans = img_trans.to(torch.float16)
                dst[i] = img_trans
            dst = dst.to(memory_format=torch.channels_last)
            return dst
        imagenet_trans.is_parallel = True
        return imagenet_trans

    def declare_state_and_memory(self, previous_state):
        h, w, c = previous_state.shape
        new_shape = (c, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)
        new_dtype = torch.float16

        new_state = replace(previous_state, shape=new_shape, dtype=new_dtype, jit_mode=False)
        mem_allocation = AllocationQuery(new_shape, new_dtype)
        return (new_state, mem_allocation)

def build_imagenet_loader(config):
    config.defrost()
    config.MODEL.NUM_CLASSES = 1000 
    config.freeze()

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    label_pipeline = [IntDecoder(), ToTensor(), Squeeze(), ToDevice(torch.device(f'cuda:{config.LOCAL_RANK}'), non_blocking=True)]

    train_decoder = RandomResizedCropRGBImageDecoder((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE))
    train_pipeline = [train_decoder, ImagenetTransform(), ToDevice(torch.device(f'cuda:{config.LOCAL_RANK}'), non_blocking=True)]
    train_pipelines = {
        'image': train_pipeline,
        'label': label_pipeline
    }

    data_loader_train = Loader(os.path.join(config.DATA.DATA_PATH,'imagenet_train.beton'), batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS,
            order=OrderOption.RANDOM if dist.get_world_size()>1 else OrderOption.QUASI_RANDOM, 
            pipelines=train_pipelines, drop_last=True, distributed=dist.get_world_size()>1, os_cache=True) 

    val_decoder = CenterCropRGBImageDecoder((config.DATA.IMG_SIZE,config.DATA.IMG_SIZE),224/256)
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
    IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
    val_pipeline = [val_decoder, ToTensor(), ToDevice(torch.device(f'cuda:{config.LOCAL_RANK}'), non_blocking=True), ToTorchImage(), NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)]
    val_pipelines = {
        'image': val_pipeline,
        'label': label_pipeline
    }

    indices = np.arange(dist.get_rank(), 50000, dist.get_world_size())
    data_loader_val = Loader(os.path.join(config.DATA.DATA_PATH,'imagenet_test.beton'), batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS,
                    order=OrderOption.SEQUENTIAL, pipelines=val_pipelines, drop_last=False, indices=indices)

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return 50000, data_loader_train, data_loader_val, mixup_fn

class CifarTransform(Operation):
    def generate_code(self):
        parallel_range = Compiler.get_iterator()
        def cifar_trans(images, dst):
            ''' 
            images - b x h x w x c
            dst - b x 32 x 32 x c
            '''
            mean = np.asarray([0.4914, 0.4822, 0.4465])
            std = np.asarray([0.2023, 0.1994, 0.2010])
            img_size = config.DATA.IMG_SIZE
            normalize = transforms.Normalize(
                mean=mean,
                std=std
            )
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                normalize
            ])
            for i in parallel_range(images.shape[0]):
                img_trans = transform(images[i])
                img_trans = img_trans.to(torch.float16)
                dst[i] = img_trans
            dst = dst.to(memory_format=torch.channels_last)
            return dst
        cifar_trans.is_parallel = True
        return cifar_trans 

    def declare_state_and_memory(self, previous_state):
        h, w, c = previous_state.shape
        new_shape = (c, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)
        new_dtype = torch.float16

        new_state = replace(previous_state, shape=new_shape, dtype=new_dtype, jit_mode=False)
        mem_allocation = AllocationQuery(new_shape, new_dtype)
        return (new_state, mem_allocation)

def build_cifar_loader(config):
    config.defrost()
    config.MODEL.NUM_CLASSES = 10
    config.freeze()

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    label_pipeline = [IntDecoder(), ToTensor(), Squeeze(), ToDevice(torch.device(f'cuda:{config.LOCAL_RANK}'), non_blocking=True)]

    train_decoder = SimpleRGBImageDecoder() 
    train_pipeline = [train_decoder, CifarTransform(), ToDevice(torch.device(f'cuda:{config.LOCAL_RANK}'), non_blocking=True)]
    train_pipelines = {
        'image': train_pipeline,
        'label': label_pipeline
    }

    data_loader_train = Loader(os.path.join(config.DATA.DATA_PATH,'cifar_train.beton'), batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS,
            order=OrderOption.RANDOM if dist.get_world_size()>1 else OrderOption.QUASI_RANDOM,
            pipelines=train_pipelines, drop_last=True, distributed=dist.get_world_size()>1, os_cache=True)

    val_decoder = SimpleRGBImageDecoder() 
    mean = np.asarray([0.4914, 0.4822, 0.4465]) * 255
    std = np.asarray([0.2023, 0.1994, 0.2010]) * 255
    val_pipeline = [val_decoder, ToTensor(), ToDevice(torch.device(f'cuda:{config.LOCAL_RANK}'), non_blocking=True), ToTorchImage(), NormalizeImage(mean, std, np.float16)]
    val_pipelines = {
        'image': val_pipeline,
        'label': label_pipeline
    }

    indices = np.arange(dist.get_rank(), 10000, dist.get_world_size())
    data_loader_val = Loader(os.path.join(config.DATA.DATA_PATH,'cifar_test.beton'), batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS,
                    order=OrderOption.SEQUENTIAL, pipelines=val_pipelines, drop_last=False, indices=indices)

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return 10000, data_loader_train, data_loader_val, mixup_fn


def build_loader(config_):
    global config
    config = config_
    if config.DATA.DATASET == 'imagenet':
        return build_imagenet_loader(config)
    else:
        return build_cifar_loader(config)
