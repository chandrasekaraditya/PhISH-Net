import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from load_raw import preprocess_depthmap, load_depthmap
from torchvision.transforms import functional as F
import numpy as np
from load_raw import load_image

class HDRDataset(Dataset):
    def __init__(self, data_dir='../data', params=None, suffix='', aug=False, split='train'):
        self.data_dir = data_dir
        self.suffix = suffix
        self.aug = aug

        self.in_files = open(f'{data_dir}/splits/{split}_list.txt', 'r').read().splitlines()

        ls = params['net_input_size']
        fs = params['net_output_size']
        self.ls, self.fs = ls, fs
        self.low = transforms.Compose([
            transforms.Resize((ls,ls), Image.BICUBIC),
            transforms.ToTensor()
        ])
        self.correction = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0),
        ])
        self.out = transforms.Compose([
            transforms.Resize((fs,fs), Image.BICUBIC),
            transforms.ToTensor()
        ])
        self.full = transforms.Compose([
            transforms.Resize((fs,fs), Image.BICUBIC),
            transforms.ToTensor()
        ])

        self.depth_trans = transforms.Compose([
            transforms.Resize((fs,fs), Image.BICUBIC),
            # transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, idx):
        fname = self.in_files[idx]
        imagein = Image.open(os.path.join(self.data_dir, 'input'+self.suffix, fname)).convert('RGB')
        imageout = Image.open(os.path.join(self.data_dir, 'gt'+self.suffix, fname)).convert('RGB')
        if self.aug:
            imagein = self.correction(imagein)
        imagein_low = self.low(imagein)
        imagein_full = self.full(imagein)
        imageout = self.out(imageout)

        im = load_image(os.path.join(self.data_dir, 'input'+self.suffix, fname), max_side=self.fs )
        depth = preprocess_depthmap(im, load_depthmap(os.path.join(self.data_dir, 'depthmaps'+self.suffix, fname), (self.fs, self.fs) ))
        depth =torch.Tensor(depth)

        return imagein_low,imagein_full,imageout, depth

    def list_files(self, in_path):
        files = []
        for (dirpath, dirnames, filenames) in os.walk(in_path):
            files.extend(filenames)
            break
        files = sorted([os.path.join(in_path, x) for x in files])
        return files
