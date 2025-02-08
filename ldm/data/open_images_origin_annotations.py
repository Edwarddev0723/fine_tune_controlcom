import os
from io import BytesIO
import logging
import base64
from sre_parse import State
from sys import prefix
import threading
import random
from turtle import left, right
from cv2 import norm
import numpy as np
from typing import Callable, List, Tuple, Union
from PIL import Image,ImageDraw
from sympy import source
import torch.utils.data as data
import json
import time
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import copy
import math
from functools import partial
import albumentations as A
import bezier
from tqdm import tqdm
import sys
import shutil

import transformers
proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, proj_dir)


def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))


def get_tensor(normalize=True, toTensor=True, resize=True, image_size=(512, 512)):
    transform_list = []
    if resize:
        transform_list += [torchvision.transforms.Resize(image_size)]
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]
    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True, resize=True, image_size=(224, 224)):
    transform_list = []
    if resize:
        transform_list += [torchvision.transforms.Resize(image_size)]
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]
    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def scan_all_files():
    bbox_dir = os.path.join(proj_dir, '../../dataset/open-images/bbox_mask')
    assert os.path.exists(bbox_dir), bbox_dir
    
    bad_files = []
    for split in os.listdir(bbox_dir):
        total_images, total_pairs, bad_masks, bad_images = 0, 0, 0, 0
        subdir = os.path.join(bbox_dir, split)
        if not os.path.isdir(subdir) or split not in ['train', 'test', 'validation']:
            continue
        for file in tqdm(os.listdir(subdir)):
            try:
                with open(os.path.join(subdir, file), 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        info = line.split(' ')
                        mask_file = os.path.join(bbox_dir, '../masks', split, info[-2])
                        if os.path.exists(mask_file):
                            total_pairs += 1
                        else:
                            bad_masks += 1
                total_images += 1
            except:
                bad_files.append(file)
                bad_images += 1
        print('{}, {} images({} bad), {} pairs({} bad)'.format(
            split, total_images, bad_images, total_pairs, bad_masks))
        
        if len(bad_files) > 0:
            with open(os.path.join(bbox_dir, 'bad_files.txt'), 'w') as f:
                for file in bad_files:
                    f.write(file + '\n')
        
    print(f'{len(bad_files)} bad_files')

class DataAugmentation:
    def __init__(self, augment_background, augment_bbox, border_mode=0):
        self.blur = A.Blur(p=0.2)
        self.appearance_trans = A.Compose([
            A.ColorJitter(brightness=0.2, 
                          contrast=0.2, 
                          saturation=0.2, 
                          hue=0.05, 
                          always_apply=False, 
                          p=0.9)],
            additional_targets={'image':'image', 'image1':'image'}
            )
        self.geometric_trans = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20,
                     border_mode=border_mode,
                     value=(127,127,127),
                     mask_value=0,
                     p=0.9),
            A.Perspective(scale=(0.05, 0.1), 
                          pad_mode=border_mode,
                          pad_val =(127,127,127),
                          mask_pad_val=0,
                          fit_output=True, 
                          p=0.9)
        ])
        self.crop_bg_p  = 0.5
        self.pad_bbox_p = 0.5 if augment_bbox else 0
        self.augment_background_p = 0.5 if augment_background else 0
        self.bbox_maxlen = 0.8
    
    def __call__(self, bg_img, bbox, bg_mask, fg_img, fg_mask, indicator, new_bg):
        # randomly crop background image
        if self.crop_bg_p > 0 and np.random.rand() < self.crop_bg_p:
            crop_bg, crop_bbox, crop_mask = self.random_crop_background(bg_img, bbox, bg_mask)
        else:
            crop_bg, crop_bbox, crop_mask = bg_img.copy(), bbox.copy(), bg_mask.copy()
        # randomly pad bounding box of foreground
        if self.pad_bbox_p > 0 and np.random.rand() < self.pad_bbox_p:
            pad_bbox = self.random_pad_bbox(crop_bbox, crop_bg.shape[1], crop_bg.shape[0])
        else:
            pad_bbox = crop_bbox.copy()
        pad_mask = bbox2mask(pad_bbox, crop_bg.shape[1], crop_bg.shape[0])
        # perform illumination transformation on background
        if self.augment_background_p > 0 and np.random.rand() < self.augment_background_p:
            trans_bg = self.appearance_trans(image=crop_bg.copy())['image']
        else:
            trans_bg = crop_bg.copy()
    
        # perform illumination and pose transformation on foreground
        trans_fg, trans_fgmask = self.augment_foreground(fg_img, fg_mask, indicator, new_bg)
        
        # generate composite by copy-and-paste foreground object
        if indicator[0] == 0:
            x1,y1,x2,y2 = crop_bbox
            origin_fg = trans_bg[y1:y2,x1:x2].copy()
            target_fg = trans_fg if indicator == (0,0) else fg_img
            fuse_fg = np.where(fg_mask[:,:,np.newaxis] > 127, target_fg, origin_fg)
            trans_bg[y1:y2,x1:x2] = fuse_fg

        trans_fg = self.blur(image=trans_fg)['image']

        return {"bg_img":   trans_bg,
                "bg_mask":  crop_mask,
                "bbox":     crop_bbox,
                "pad_bbox": pad_bbox,
                "pad_mask": pad_mask,
                "fg_img":   trans_fg,
                "fg_mask":  trans_fgmask,
                "gt_fg_mask": fg_mask}
    
    def augment_foreground(self, image, mask, indicator, new_bg):
        trans_img, trans_mask = copy.deepcopy(image), copy.deepcopy(mask)

        if indicator == (1,0) or indicator == (0,0):
            # appearance transformed image
            transformed = self.appearance_trans(image=trans_img)
            trans_img   = transformed['image']
        elif indicator == (0,1):
            # geometric transformed image
            transformed = self.geometric_trans(image=trans_img, mask=trans_mask)
            trans_img  = transformed['image']
            trans_mask = transformed['mask']
        elif indicator == (1,1):
            if new_bg is None:
                transformed = self.appearance_trans(image=trans_img)
            else:
                transformed = self.appearance_trans(image=trans_img, image1=new_bg)
                new_bg = transformed['image1']
            transformed = self.geometric_trans(image=transformed['image'], mask=trans_mask)
            trans_img  = transformed['image']
            trans_mask = transformed['mask']
        if indicator[-1] == 1 and new_bg is not None:
            trans_img = np.where(trans_mask[:,:,np.newaxis] > 127, trans_img, new_bg)
        return trans_img, trans_mask

    def random_crop_background(self, image, bbox, mask):
        trans_bbox  = copy.deepcopy(bbox)
        trans_image = image.copy()
        trans_mask  = mask.copy() 
        
        width, height = image.shape[1], image.shape[0]
        bbox_w = float(bbox[2] - bbox[0]) / width
        bbox_h = float(bbox[3] - bbox[1]) / height
        
        left, right, top, down = 0, width, 0, height 
        if bbox_w < self.bbox_maxlen:
            maxcrop = (width - bbox_w * width / self.bbox_maxlen) / 2
            left  = int(np.random.rand() * min(maxcrop, bbox[0]))
            right = width - int(np.random.rand() * min(maxcrop, width - bbox[2]))

        if bbox_h < self.bbox_maxlen:
            maxcrop = (height - bbox_h * height / self.bbox_maxlen) / 2
            top   = int(np.random.rand() * min(maxcrop, bbox[1]))
            down  = height - int(np.random.rand() * min(maxcrop, height - bbox[3]))
        
        trans_bbox = [bbox[0] - left, bbox[1] - top, bbox[2] - left, bbox[3] - top]
        trans_image = trans_image[top:down, left:right]
        trans_mask  = trans_mask[top:down, left:right]
        # print(image.shape, trans_image.shape, trans_mask.shape, bbox, trans_bbox)
        return trans_image, trans_bbox, trans_mask
    
    def random_pad_bbox(self, bbox, width, height):
        bbox_pad  = copy.deepcopy(bbox)
        bbox_w = float(bbox[2] - bbox[0]) / width
        bbox_h = float(bbox[3] - bbox[1]) / height
        
        if bbox_w < self.bbox_maxlen:
            maxpad = width * min(self.bbox_maxlen - bbox_w, bbox_w) * 0.5
            bbox_pad[0] = max(0, int(bbox[0] - np.random.rand() * maxpad))
            bbox_pad[2] = min(width, int(bbox[2] + np.random.rand() * maxpad))
        
        if bbox_h < self.bbox_maxlen:
            maxpad = height * min(self.bbox_maxlen - bbox_h, bbox_h) * 0.5
            bbox_pad[1] = max(0, int(bbox[1] - np.random.rand() * maxpad))
            bbox_pad[3] = min(height, int(bbox[3] + np.random.rand() * maxpad))
        return bbox_pad

    
def bbox2mask(bbox, mask_w, mask_h):
    mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
    mask[bbox[1] : bbox[3], bbox[0] : bbox[2]] = 255
    return mask
    
def mask2bbox(mask):
    if not isinstance(mask, np.ndarray):
        mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = np.squeeze(mask, axis=-1)
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [x1, y1, x2, y2]

    
def constant_pad_bbox(bbox, width, height, value=10):
    ### Get reference image
    bbox_pad=copy.copy(bbox)
    left_space  = bbox[0]
    up_space    = bbox[1]
    right_space = width  - bbox[2]
    down_space  = height - bbox[3] 

    bbox_pad[0]=bbox[0]-min(value, left_space)
    bbox_pad[1]=bbox[1]-min(value, up_space)
    bbox_pad[2]=bbox[2]+min(value, right_space)
    bbox_pad[3]=bbox[3]+min(value, down_space)
    return bbox_pad
    
    
def crop_image_by_bbox(img, bbox, pad_bbox=10):
    if isinstance(img, np.ndarray):
        width,height = img.shape[1], img.shape[0]
    else:
       width,height = img[0].shape[1], img[0].shape[0]
    bbox_pad = constant_pad_bbox(bbox, width, height, pad_bbox) if pad_bbox > 0 else bbox
    if isinstance(img, (list, tuple)):
        crop = [per_img[bbox_pad[1]:bbox_pad[3],bbox_pad[0]:bbox_pad[2]].copy() for per_img in img]
    else:
        crop = img[bbox_pad[1]:bbox_pad[3],bbox_pad[0]:bbox_pad[2]].copy()
    return crop, bbox_pad

def image2inpaint(image, mask):
    if len(mask.shape) == 2:
        mask_f = mask[:,:,np.newaxis].copy()
    else:
        mask_f = mask.copy()
    mask_f  = mask_f.astype(np.float32) / 255
    inpaint = image.astype(np.float32)
    gray  = np.ones_like(inpaint) * 127
    inpaint = inpaint * (1 - mask_f) + mask_f * gray
    inpaint = np.uint8(inpaint)
    return inpaint

def check_dir(dir):
    assert os.path.exists(dir), dir
    return dir

def get_bbox_tensor(bbox, width, height):
    norm_bbox = copy.deepcopy(bbox)
    norm_bbox = torch.tensor(norm_bbox).reshape(-1).float()
    norm_bbox[0::2] /= width
    norm_bbox[1::2] /= height
    return norm_bbox
    
def reverse_image_tensor(tensor, img_size=(256,256)):
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = (tensor.float() + 1) / 2
    tensor = torch.clamp(tensor, min=0.0, max=1.0)
    tensor = torch.permute(tensor, (0, 2, 3, 1)) * 255
    tensor = tensor.detach().cpu().numpy()
    img_nps = np.uint8(tensor)
    def np2bgr(img, img_size = img_size):
        if img.shape[:2] != img_size:
            img = cv2.resize(img, img_size)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_list = [np2bgr(img) for img in img_nps]
    return img_list

def reverse_mask_tensor(tensor, img_size=(256,256)):
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = torch.clamp(tensor, min=0.0, max=1.0)
    tensor = torch.permute(tensor.float(), (0, 2, 3, 1)) * 255
    tensor = tensor.detach().cpu().numpy()
    img_nps = np.uint8(tensor)
    def np2bgr(img, img_size = img_size):
        if img.shape[:2] != img_size:
            img = cv2.resize(img, img_size)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_list = [np2bgr(img) for img in img_nps]
    return img_list

def reverse_clip_tensor(tensor, img_size=(256,256)):
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073],  dtype=torch.float)
    MEAN = MEAN.reshape(1, 3, 1, 1).to(tensor.device)
    STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float)
    STD  = STD.reshape(1, 3, 1, 1).to(tensor.device)
    tensor = (tensor * STD) + MEAN
    tensor = torch.clamp(tensor, min=0.0, max=1.0)
    tensor = torch.permute(tensor.float(), (0, 2, 3, 1)) * 255
    tensor = tensor.detach().cpu().numpy()
    img_nps = np.uint8(tensor)
    def np2bgr(img, img_size = img_size):
        if img.shape[:2] != img_size:
            img = cv2.resize(img, img_size)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_list = [np2bgr(img) for img in img_nps]
    return img_list

def random_crop_image(image, crop_w, crop_h):
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)
    x_space = image.shape[1] - crop_w
    y_space = image.shape[0] - crop_h
    x1 = np.random.randint(0, x_space) if x_space > 0 else 0
    y1 = np.random.randint(0, y_space) if y_space > 0 else 0
    crop = image[y1 : y1+crop_h, x1 : x1+crop_w]
    return crop
    

class OpenImageDataset(data.Dataset):
    def __init__(self,split,**args):
        self.split=split
        dataset_dir = args['dataset_dir']
        self.parse_augment_config(args)
        assert os.path.exists(dataset_dir), dataset_dir
        self.bbox_dir = check_dir(os.path.join(dataset_dir, 'bbox_mask', split))
        self.image_dir= check_dir(os.path.join(dataset_dir, 'images', split))
        self.mask_dir = check_dir(os.path.join(dataset_dir, 'masks', split))
        self.bbox_path_list = self.load_bbox_path_list()
        self.length=len(self.bbox_path_list)
        self.random_trans = DataAugmentation(self.augment_background, self.augment_box)
        self.clip_transform = get_tensor_clip(image_size=(224, 224))
        self.image_size = args['image_size'], args['image_size']
        self.sd_transform = get_tensor(image_size=self.image_size)
        self.mask_transform = get_tensor(normalize=False, image_size=self.image_size)
        self.clip_mask_transform = get_tensor(normalize=False, image_size=(16, 16))
    
    def load_bbox_path_list(self):
        cache_dir  = self.bbox_dir
        cache_file = os.path.join(cache_dir, f'{self.split}.json')
        if os.path.exists(cache_file):
            print('load bbox list from ', cache_file)
            with open(cache_file, 'r') as f:
                bbox_path_list = json.load(f)
        else:
            bbox_path_list= os.listdir(self.bbox_dir)
            bbox_path_list.sort()
            print('save bbox list to ', cache_file)
            with open(cache_file, 'w') as f:
                json.dump(bbox_path_list, f)
        return bbox_path_list

    
    def parse_augment_config(self, args):
        self.augment_config = args['augment_config'] if 'augment_config' in args else None
        if self.augment_config:
            self.sample_mode = self.augment_config.sample_mode
            self.augment_types = self.augment_config.augment_types
            self.sample_prob = self.augment_config.sample_prob
            # print('data_sample_mode:{}, augment_types:{}, sample_prob:{}'.format(
            #     self.sample_mode, self.augment_types, self.sample_prob
            # ))
            if self.sample_mode == 'random':
                assert len(self.augment_types) == len(self.sample_prob), \
                    'len({}) != len({})'.format(self.augment_types, self.sample_prob)
            self.augment_background = self.augment_config.augment_background
            self.augment_box = self.augment_config.augment_box
            self.replace_background_prob = self.augment_config.replace_background_prob
        else:
            self.sample_mode   = 'random'
            self.augment_types = [(0,0), (0,1), (1,0), (1,1)]
            self.sample_prob = [1. / len(self.augment_types)] * len(self.augment_types)
            self.augment_background = False
            self.augment_box = False
            self.replace_background_prob = 0.5

    def load_bbox_file(self, bbox_file):
        bbox_list = []
        with open(bbox_file) as f:
            for line in f.readlines():
                info  = line.strip().split(' ')
                label = info[-3]
                confidence = float(info[-1])
                bbox  = [int(float(f)) for f in info[:4]]
                mask  = os.path.join(self.mask_dir, info[-2])
                if os.path.exists(mask):
                    bbox_list.append((bbox, label, mask, confidence))
        return bbox_list
    
    def sample_all_augmentations(self, source_np, bbox, mask, fg_img, fg_mask, new_bg):
        output = {}
        for indicator in self.augment_types:
            sample = self.sample_one_augmentations(source_np, bbox, mask, fg_img, fg_mask, indicator, new_bg)
            for k,v in sample.items():
                if k not in output:
                    output[k] = [v]
                else:
                    output[k].append(v)
        for k in output.keys():
            output[k] = torch.stack(output[k], dim=0)
        return output
    
    def sample_one_augmentations(self, source_np, bbox, mask, fg_img, fg_mask, indicator, new_bg):
        transformed = self.random_trans(source_np, bbox, mask, fg_img, fg_mask, indicator, new_bg)
        # get background and bbox
        img_width, img_height = transformed["bg_mask"].shape[1], transformed["bg_mask"].shape[0]
        gt_mask_tensor = self.mask_transform(Image.fromarray(transformed["bg_mask"]))
        gt_mask_tensor = torch.where(gt_mask_tensor > 0.5, 1, 0).float()
        gt_img_tensor  = self.sd_transform(Image.fromarray(transformed['bg_img']))
        gt_bbox_tensor = get_bbox_tensor(transformed['bbox'], img_width, img_height)
        mask_tensor = self.mask_transform(Image.fromarray(transformed['pad_mask']))
        mask_tensor = torch.where(mask_tensor > 0.5, 1, 0).float()
        bbox_tensor = get_bbox_tensor(transformed['pad_bbox'], img_width, img_height)
        # get foreground and foreground mask
        fg_mask_tensor = self.clip_mask_transform(Image.fromarray(transformed['fg_mask']))
        fg_mask_tensor = torch.where(fg_mask_tensor > 0.5, 1, 0)
        gt_fg_mask_tensor = self.clip_mask_transform(Image.fromarray(transformed['gt_fg_mask']))
        gt_fg_mask_tensor = torch.where(gt_fg_mask_tensor > 0.5, 1, 0)
        fg_img_tensor = self.clip_transform(Image.fromarray(transformed['fg_img']))
        indicator_tensor = torch.tensor(indicator, dtype=torch.int32)
        inpaint = gt_img_tensor * (1 - mask_tensor)
        return {"gt_img":  gt_img_tensor,
                "gt_mask": gt_mask_tensor,
                "gt_bbox": gt_bbox_tensor,
                "bg_img": inpaint,
                "bg_mask": mask_tensor,
                "fg_img":  fg_img_tensor,
                "fg_mask": fg_mask_tensor,
                "gt_fg_mask": gt_fg_mask_tensor,
                "bbox": bbox_tensor,
                "indicator": indicator_tensor}

    def replace_background_in_foreground(self, fg_img, fg_mask, index):
        bg_idx = int((np.random.randint(1, 100) + index) % self.length)
        bbox_file = self.bbox_path_list[bg_idx]
        # get source image and mask
        image_path = os.path.join(self.image_dir, os.path.splitext(bbox_file)[0] + '.jpg')
        bg_img = Image.open(image_path).convert("RGB")
        bg_width, bg_height = bg_img.size
        fg_height, fg_width = fg_img.shape[:2]
        if bg_width < fg_width or bg_height < fg_height:
            scale = max(float(fg_width) / bg_width, float(fg_height) / bg_height)            
            bg_width  = int(scale * bg_width)
            bg_height = int(scale * bg_height)
            bg_img = bg_img.resize((bg_width, bg_height))
        bg_crop = random_crop_image(bg_img, fg_width, fg_height)
        new_fg  = np.where(fg_mask[:,:,np.newaxis] >= 127, fg_img, bg_crop)
        return new_fg, bg_crop
    
    def __getitem__(self, index):
        try:
            # get bbox and mask
            bbox_file = self.bbox_path_list[index] 
            bbox_path = os.path.join(self.bbox_dir, bbox_file)
            bbox_list  = self.load_bbox_file(bbox_path)
            bbox,label,mask_path,mask_conf = random.choice(bbox_list)
            # get source image and mask
            image_path = os.path.join(self.image_dir, os.path.splitext(bbox_file)[0] + '.jpg')
            source_img = Image.open(image_path).convert("RGB")
            source_np  = np.asarray(source_img)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (source_np.shape[1], source_np.shape[0]))
            bbox = mask2bbox(mask)
            [fg_img, fg_mask], bbox  = crop_image_by_bbox([source_np, mask], bbox)
            if self.replace_background_prob > 0 and np.random.rand() < self.replace_background_prob:
                fg_img, new_bg = self.replace_background_in_foreground(fg_img, fg_mask, index)
            else:
                new_bg = None
            bbox_mask = bbox2mask(bbox, mask.shape[1], mask.shape[0])
            mask = np.where(bbox_mask > 127, mask, bbox_mask) 
            # perform data augmentation
            if self.sample_mode == 'random':
                augment_list = list(range(len(self.augment_types)))
                if max(self.sample_prob) == 1:
                    augment_type = self.sample_prob.index(1)
                else:
                    augment_type = np.random.choice(augment_list, 1, p=self.sample_prob)[0]
                    augment_type = int(augment_type)
                indicator = self.augment_types[augment_type]
                sample = self.sample_one_augmentations(source_np, bbox, mask, fg_img, fg_mask, indicator, new_bg)
            else:
                sample = self.sample_all_augmentations(source_np, bbox, mask, fg_img, fg_mask, new_bg)
            sample['image_path'] = image_path
            return sample
        except:
            idx = np.random.randint(0, len(self)-1)
            return self[idx]
        
    def __len__(self):
        return self.length
    
class COCOEEDataset(data.Dataset):
    def __init__(self, **args):
        dataset_dir = args['dataset_dir']
        assert os.path.exists(dataset_dir), dataset_dir
        self.src_dir = os.path.join(dataset_dir, "GT_3500")
        self.ref_dir = os.path.join(dataset_dir, 'Ref_3500')
        self.mask_dir = os.path.join(dataset_dir, 'Mask_bbox_3500') 
        self.image_list = os.listdir(self.src_dir)
        self.image_list.sort()
        
        self.clip_transform = get_tensor_clip(image_size=(224, 224))
        self.image_size = args['image_size'], args['image_size']
        self.sd_transform   = get_tensor(image_size=self.image_size)
        self.mask_transform = get_tensor(normalize=False, image_size=self.image_size)
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image = self.image_list[index]
        src_path = os.path.join(self.src_dir, image)
        src_img = Image.open(src_path).convert("RGB")
        src_tensor = self.sd_transform(src_img)
        im_name  = os.path.splitext(image)[0].split('_')[0]
        ref_path = os.path.join(self.ref_dir, im_name + '_ref.png')
        assert os.path.exists(ref_path), ref_path
        ref_img = Image.open(ref_path).convert('RGB')
        ref_tensor = self.clip_transform(ref_img)
        mask_path = os.path.join(self.mask_dir, im_name + '_mask.png')
        assert os.path.exists(mask_path), mask_path
        mask_img = Image.open(mask_path).convert("L")
        mask_img = mask_img.resize((src_img.width, src_img.height))
        bbox = mask2bbox(np.asarray(mask_img))
        bbox_tensor = get_bbox_tensor(bbox, src_img.width, src_img.height)
        mask_tensor = self.mask_transform(mask_img) 
        mask_tensor = torch.where(mask_tensor > 0.5, 1, 0).float()
        indicator_tensor = torch.tensor([1,1], dtype=torch.int32)
        inpaint = src_tensor * (1 - mask_tensor)
        
        return {"image_path": src_path,
                "gt_img":  src_tensor,
                "bg_img":  inpaint,
                "bg_mask": mask_tensor,
                "fg_img":  ref_tensor,
                "bbox":    bbox_tensor,
                "indicator": indicator_tensor}
    
def vis_all_augtypes(batch):
    file = batch['image_path']
    gt_t = batch['gt_img'][0]
    gtmask_t = batch['gt_mask'][0]
    bg_t = batch['bg_img'][0]
    bgmask_t  = batch['bg_mask'][0]
    fg_t = batch['fg_img'][0]
    fgmask_t  = batch['fg_mask'][0]
    indicator = batch['indicator'][0].numpy()
    
    gt_imgs  = reverse_image_tensor(gt_t)
    gt_masks = reverse_mask_tensor(gtmask_t) 
    bg_imgs  = reverse_image_tensor(bg_t)
    bg_masks = reverse_mask_tensor(bgmask_t)
    fg_imgs  = reverse_clip_tensor(fg_t)
    fg_masks = reverse_mask_tensor(fgmask_t)
    
    ver_border = np.ones((gt_imgs[0].shape[0], 10, 3), dtype=np.uint8) * np.array([0, 0, 250]).reshape((1,1,-1))
    img_list = []
    for i in range(len(fg_imgs)):
        text = '[{},{}]'.format(indicator[i][0], indicator[i][1])
        fg_img = fg_imgs[i].copy()
        cv2.putText(fg_img, text, (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
        cat_img = np.concatenate([gt_imgs[i], ver_border, gt_masks[i], ver_border, bg_imgs[i], 
                                  ver_border, fg_img, ver_border, fg_masks[i]], axis=1)
        if i > 0:
            hor_border = np.ones((10, cat_img.shape[1], 3), dtype=np.uint8) * np.array([0, 0, 250]).reshape((1,1,-1))
            img_list.append(hor_border)
        img_list.append(cat_img)
    img_batch = np.concatenate(img_list, axis=0)
    return img_batch

def vis_random_augtype(batch):
    file = batch['image_path']
    gt_t = batch['gt_img']
    gtmask_t = batch['gt_mask']
    bg_t = batch['bg_img']
    bgmask_t  = batch['bg_mask']
    fg_t = batch['fg_img']
    fgmask_t = batch['fg_mask']
    gt_fgmask_t = batch['gt_fg_mask']
    indicator = batch['indicator'].numpy()
    
    gt_imgs  = reverse_image_tensor(gt_t)
    gt_masks = reverse_mask_tensor(gtmask_t) 
    bg_imgs  = reverse_image_tensor(bg_t)
    bg_masks = reverse_mask_tensor(bgmask_t)
    fg_imgs  = reverse_clip_tensor(fg_t)
    fg_masks = reverse_mask_tensor(fgmask_t)
    gt_fgmasks = reverse_mask_tensor(gt_fgmask_t)

    ver_border = np.ones((gt_imgs[0].shape[0], 10, 3), dtype=np.uint8) * np.array([0, 0, 250]).reshape((1,1,-1))
    img_list = []
    for i in range(len(gt_imgs)):
        im_name = os.path.basename(file[i]) if len(file) > 1 else os.path.basename(file[0])
        text = '[{},{}]'.format(indicator[i][0], indicator[i][1])
        fg_img = fg_imgs[i].copy()
        cv2.putText(fg_img, text, (10,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cat_img = np.concatenate([gt_imgs[i], ver_border, gt_masks[i], ver_border, bg_imgs[i], 
                                  ver_border, fg_img, ver_border, fg_masks[i], gt_fgmasks[i]], axis=1)
        if i > 0:
            hor_border = np.ones((10, cat_img.shape[1], 3), dtype=np.uint8) * np.array([0, 0, 250]).reshape((1,1,-1))
            img_list.append(hor_border)
        img_list.append(cat_img)
    img_batch = np.concatenate(img_list, axis=0)
    return img_batch

from kornia.filters import gaussian_blur2d

def fill_mask(mask, bbox, kernel_size, sigma):
    x1, y1, x2, y2 = bbox
    out_mask = copy.deepcopy(mask)
    local_mask = out_mask[:, :, y1:y2, x1:x2]
    print('local', local_mask.shape, kernel_size, sigma)
    local_mask = gaussian_blur2d(local_mask, kernel_size, sigma) # border_type='constant'
    local_mask = torch.where(local_mask > 1e-5, 1., 0.).float()
    out_mask[:, :, y1:y2, x1:x2] = local_mask
    return out_mask

def compute_mask_iou(
    mask1: torch.Tensor,
    mask2: torch.Tensor,
    ) -> torch.Tensor:
    """
    Inputs:
    mask1: NxHxW torch.float32. Consists of [0, 1]
    mask2: NxHxW torch.float32. Consists of [0, 1]
    Outputs:
    ret: NxM torch.float32. Consists of [0 - 1]
    """
    if mask1.ndim == 4:
        mask1.squeeze_(1)
    if mask2.ndim == 4:
        mask2.squeeze_(1)
    N, H, W = mask1.shape
    M, H, W = mask2.shape

    mask1 = mask1.view(N, H*W)
    mask2 = mask2.view(M, H*W)

    intersection = torch.matmul(mask1, mask2.t())

    area1 = mask1.sum(dim=1).view(1, -1)
    area2 = mask2.sum(dim=1).view(1, -1)

    union = (area1.t() + area2) - intersection

    ret = torch.where(
        union == 0,
        torch.tensor(0., device=mask1.device),
        intersection / union,
    )
    return ret.mean()

def check_kernel_size(ks, min_ks, max_ks):
    if int(ks) & 1 == 0:
        ks = int(ks) + 1
    ks = int(min(max(min_ks, ks), max_ks))
    return ks

def get_mask_coef(x, x0=0.8, y0=0.2):
    if x < x0:
        return x / x0 * y0
    else:
        return y0 + (x - x0) / (1 - x0) * (1 - y0)
    
    
def test_fill_mask(batch, index):
    vis_dir = os.path.join(proj_dir, 'outputs/mask_gaussian_blur')
    os.makedirs(vis_dir, exist_ok=True)
    
    gtmask_t  = batch['gt_mask']
    bgmask_t  = 1 - batch['bg_mask']
    gtbbox_norm = batch['gt_bbox']
    gtbbox_int  = (gtbbox_norm * gtmask_t.shape[-1]).int()
    bbox_norm = batch["bbox"]
    bbox_int  = (bbox_norm * gtmask_t.shape[-1]).int()
    sigma_cofs  = np.linspace(0,1,5).tolist()
    sigma_list  = [[] for _ in sigma_cofs]
    kernel_list = [[] for _ in sigma_cofs] 
    padmasks = [[] for _ in sigma_cofs]
    for i in range(gtmask_t.shape[0]):
        bbox_w, bbox_h = (bbox_int[i,2] - bbox_int[i,0]).item(), (bbox_int[i,3] - bbox_int[i,1]).item()
        min_kernel = (1,1)
        max_kernel = (int(bbox_h * 2 - 1), int(bbox_w * 2 - 1))
        
        for j,sigma_cof in enumerate(sigma_cofs):
            scale_cof = get_mask_coef(sigma_cof)
            kernel_h = check_kernel_size(scale_cof * max_kernel[0], min_kernel[0], max_kernel[0])
            kernel_w = check_kernel_size(scale_cof * max_kernel[1], min_kernel[1], max_kernel[1])
            kernel_size = (kernel_h, kernel_w)
            sigma = (kernel_h / 3., kernel_w / 3.)
            print(sigma, kernel_size, bbox_w, bbox_h)
            padmasks[j].append(fill_mask(gtmask_t[i:i+1], bbox_int[i], kernel_size, sigma))
            sigma_list[j].append(sigma)
            kernel_list[j].append(kernel_size)

    padmasks = [torch.cat(t, dim=0) for t in padmasks]
    padmasks = [reverse_mask_tensor(t) for t in padmasks]
    gtmasks  = reverse_mask_tensor(gtmask_t)
    bgmasks  = reverse_mask_tensor(bgmask_t)
    img_list = []
    ver_border = np.ones((gtmasks[0].shape[0], 10, 3), dtype=np.uint8) * np.array([0, 0, 250]).reshape((1,1,-1))
    for i in range(len(gtmasks)):
        x1,y1,x2,y2 = gtbbox_int[i].numpy().tolist()
        src = gtmasks[i]
        src = cv2.rectangle(src, (x1, y1), (x2, y2), color=(0, 0, 250), thickness=3)
        bbox_w, bbox_h = bbox_int[i,2] - bbox_int[i,0], bbox_int[i,3] - bbox_int[i,1]
        cat_img = [src]
        for j in range(len(sigma_cofs)):
            pad = padmasks[j][i]
            text = 's:{}, kernel:{}'.format(sigma_cofs[j], kernel_list[j][i])
            cv2.putText(pad, text, (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
            cat_img.append(ver_border)
            cat_img.append(pad)
            # pad = cv2.rectangle(pad, (x1, y1), (x2, y2), color=(0, 0, 250), thickness=3)
        dst = bgmasks[i]
        dst = cv2.rectangle(dst, (x1, y1), (x2, y2), color=(0, 0, 250), thickness=3)
        cat_img.append(ver_border)
        cat_img.append(dst)
        cat_img = np.concatenate(cat_img, axis=1)
        # cat_img = np.concatenate([src, ver_border, pad, ver_border, dst], axis=1)
        if i > 0:
            hor_border = np.ones((10, cat_img.shape[1], 3), dtype=np.uint8) * np.array([0, 0, 250]).reshape((1,1,-1))
            img_list.append(hor_border)
        img_list.append(cat_img)
    batch_img = np.concatenate(img_list, axis=0)
    cv2.imwrite(os.path.join(vis_dir, f'batch{index}.jpg'), batch_img)
    
def test_mask_blur_batch():
    vis_dir = os.path.join(proj_dir, 'outputs/mask_blur_linear')
    os.makedirs(vis_dir, exist_ok=True)
    
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from torch.utils.data import DataLoader
    cfg_path = os.path.join(proj_dir, 'configs/v1.yaml')
    configs  = OmegaConf.load(cfg_path).data.params.validation
    dataset  = instantiate_from_config(configs)
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=4, 
                            shuffle=False,
                            num_workers=4)
    print('{} samples = {} bs x {} batches'.format(
        len(dataset), dataloader.batch_size, len(dataloader)
    ))
    
    from ldm.modules.mask_blur import GaussianBlurMask
    device = torch.device('cuda:0')
    mask_blur = GaussianBlurMask()
    for i, batch in enumerate(dataloader):
        gtmask_t  = batch['gt_mask'].to(device)
        bgmask_t  = 1 - batch['bg_mask'].to(device)
        bbox_norm = batch['bbox'].to(device)
        t = torch.randint(0, 1000, (gtmask_t.shape[0],), device=device).long()
        masks1 = mask_blur(gtmask_t, bbox_norm, t)
        print(masks1.shape)
        masks2 = mask_blur(gtmask_t, bbox_norm, t-50)
        
        src_masks  = reverse_mask_tensor(gtmask_t)
        dst_masks1 = reverse_mask_tensor(masks1)
        dst_masks2 = reverse_mask_tensor(masks2)
        ver_border = np.ones((src_masks[0].shape[0], 10, 3), dtype=np.uint8) * np.array([0, 0, 250]).reshape((1,1,-1))
        img_list = []
        for j in range(masks1.shape[0]):
            text = 't:{}'.format(t[j])
            cv2.putText(dst_masks1[j], text, (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
            text = 't:{}'.format(t[j]-50)
            cv2.putText(dst_masks2[j], text, (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
            cat_img = np.concatenate([src_masks[j], ver_border, dst_masks1[j], ver_border, dst_masks2[j]], axis=1)
            if j > 0:
                hor_border = np.ones((10, cat_img.shape[1], 3), dtype=np.uint8) * np.array([0, 0, 250]).reshape((1,1,-1))
                img_list.append(hor_border)
            img_list.append(cat_img)
        batch_img = np.concatenate(img_list, axis=0)
        cv2.imwrite(os.path.join(vis_dir, f'batch{i}.jpg'), batch_img)
        if i > 5:
            break
    
def test_cocoee_dataset():
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from torch.utils.data import DataLoader
    cfg_path = os.path.join(proj_dir, 'configs/v1.yaml')
    configs  = OmegaConf.load(cfg_path).data.params.test
    dataset  = instantiate_from_config(configs)
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=4, 
                            shuffle=False,
                            num_workers=4)
    print('{} samples = {} bs x {} batches'.format(
        len(dataset), dataloader.batch_size, len(dataloader)
    ))
    for i, batch in enumerate(dataloader):
        file = batch['image_path']
        gt_t = batch['gt_img']
        bgmask_t  = batch['bg_mask']
        fg_t = batch['fg_img']
        bbox_t = batch['bbox']
        im_name = os.path.basename(file[0])
        # test_fill_mask(batch, i)
        print(i, len(dataloader), gt_t.shape, fg_t.shape, gt_t.shape, bbox_t.shape)
    

def test_open_images():
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from torch.utils.data import DataLoader
    cfg_path = os.path.join(proj_dir, 'configs/finetune_paint.yaml')
    configs  = OmegaConf.load(cfg_path).data.params.train
    configs.params.split = 'validation'
    configs.params.augment_config.sample_mode = 'all' 
    configs.params.augment_config.augment_types = [(0,0), (1,0), (0,1), (1,1)]
    aug_cfg  = configs.params.augment_config
    dataset  = instantiate_from_config(configs)
    bs = 1 if aug_cfg.sample_mode == 'all' else 4
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=bs, 
                            shuffle=False,
                            num_workers=0)
    print('{} samples = {} bs x {} batches'.format(
        len(dataset), dataloader.batch_size, len(dataloader)
    ))
    vis_dir = os.path.join(proj_dir, 'outputs/test_dataaug/batch_data')
    if os.path.exists(vis_dir):
        shutil.rmtree(vis_dir)
    os.makedirs(vis_dir, exist_ok=True)
    
    for i, batch in enumerate(dataloader):
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor) and batch[k].shape[0] == 1:
                batch[k] = batch[k][0]

        file = batch['image_path']
        gt_t = batch['gt_img']
        gtmask_t = batch['gt_mask']
        bgmask_t  = batch['bg_mask']
        fg_t = batch['fg_img']
        bbox_t = batch['bbox']
        im_name = os.path.basename(file[0])
        # test_fill_mask(batch, i)
        print(i, len(dataloader), gt_t.shape, gtmask_t.shape, fg_t.shape, gt_t.shape, bbox_t.shape)
        batch_img = vis_random_augtype(batch)
        cv2.imwrite(os.path.join(vis_dir, f'batch{i}.jpg'), batch_img)
        if i > 10:
            break
    
def test_open_images_efficiency():
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from torch.utils.data import DataLoader
    cfg_path = os.path.join(proj_dir, 'configs/finetune_paint.yaml')
    configs  = OmegaConf.load(cfg_path).data.params.train
    dataset  = instantiate_from_config(configs)
    bs = 16
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=bs, 
                            shuffle=False,
                            num_workers=16)
    print('{} samples = {} bs x {} batches'.format(
        len(dataset), dataloader.batch_size, len(dataloader)
    ))
    start = time.time()
    for i,batch in enumerate(dataloader):
        image = batch['gt_img']
        end = time.time()
        if i % 10 == 0:
            print('{}, avg time {:.1f}ms'.format(
                i, (end-start) / (i+1) * 1000
            ))
        
if __name__ == '__main__':
    # test_mask_blur_batch()
    # test_open_images()
    test_open_images_efficiency()
    
    
    


