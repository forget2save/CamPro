import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision as tv
import os.path as osp
import random
from torch.utils.data import Dataset, DataLoader, random_split
from glob import glob
from PIL import Image


class FullPatchDataset(Dataset):
    def __init__(self, root, psize=512) -> None:
        super().__init__()
        self.imgs = sorted(glob(osp.join(root, "*.jp*g")))
        self.totensor = tv.transforms.ToTensor()
        self.resize = tv.transforms.Resize(psize)
        self.psize = psize

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert("RGB")
        w, h = img.size
        if min(h, w) < self.psize:
            img = self.totensor(self.resize(img))
        else:
            img = self.totensor(img)
        _, h, w = img.shape
        
        i = random.randint(0, h - self.psize)
        j = random.randint(0, w - self.psize)
        patch = img[..., i : i + self.psize, j : j + self.psize]

        if random.uniform(0, 1) < 0.5:
            patch = torch.flip(patch, dims=(-1,))
        k = random.randint(0, 3)
        patch = torch.rot90(patch, k=k, dims=(-2, -1))

        return patch


class FullPatchDatasetV2(Dataset):
    def __init__(self, root, psize=512, aug=lambda x:x) -> None:
        super().__init__()
        self.imgs = sorted(glob(osp.join(root, "*.jp*g")))
        self.totensor = tv.transforms.Compose([
            tv.transforms.Resize((psize, psize)),
            tv.transforms.ToTensor(),
        ])
        self.aug = aug
        self.psize = psize

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert("RGB")
        aug_param = tv.transforms.RandomResizedCrop.get_params(img, (0.25, 1.0), (0.75, 1.33))
        patch = tv.transforms.functional.crop(img, *aug_param)
        patch = self.totensor(patch)
        patch = self.aug(patch)
        if random.uniform(0, 1) < 0.5:
            patch = torch.flip(patch, dims=(-1,))
        return patch


class MaskPatchDataset(Dataset):
    def __init__(self, root, mask_root, psize=512) -> None:
        super().__init__()
        self.imgs = sorted(glob(osp.join(root, "*.jp*g")))
        self.masks = sorted(glob(osp.join(mask_root, "*.jp*g")))
        assert len(self.imgs) == len(self.masks)
        self.totensor = tv.transforms.ToTensor()
        self.resize = tv.transforms.Resize(psize)
        self.psize = psize

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("RGB")
        
        w, h = img.size
        if min(h, w) < self.psize:
            img = self.totensor(self.resize(img))
            mask = self.totensor(self.resize(mask))
        else:
            img = self.totensor(img)
            mask = self.totensor(mask)
        _, h, w = img.shape

        i = random.randint(0, h - self.psize)
        j = random.randint(0, w - self.psize)
        patch = img[..., i : i + self.psize, j : j + self.psize]
        patch_mask = mask[..., i : i + self.psize, j : j + self.psize]

        if random.uniform(0, 1) < 0.5:
            patch = torch.flip(patch, dims=(-1,))
            patch_mask = torch.flip(patch_mask, dims=(-1,))
        k = random.randint(0, 3)
        patch = torch.rot90(patch, k=k, dims=(-2, -1))
        patch_mask = torch.rot90(patch_mask, k=k, dims=(-2, -1))

        return patch, patch_mask


class MaskPatchDatasetV2(Dataset):
    def __init__(self, root, mask_root, psize=512) -> None:
        super().__init__()
        self.imgs = sorted(glob(osp.join(root, "*.jp*g")))
        self.masks = sorted(glob(osp.join(mask_root, "*.jp*g")))
        assert len(self.imgs) == len(self.masks)
        self.totensor = tv.transforms.Compose([
            tv.transforms.Resize((psize, psize)),
            tv.transforms.ToTensor(),
        ])
        self.colorjitter = tv.transforms.ColorJitter(0.2, 0.2)
        self.psize = psize

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("RGB")
        aug_param = tv.transforms.RandomResizedCrop.get_params(img, (0.25, 1.0), (0.75, 1.33))
        patch = tv.transforms.functional.crop(img, *aug_param)
        patch_mask = tv.transforms.functional.crop(mask, *aug_param)
        patch = self.totensor(patch)
        patch_mask = self.totensor(patch_mask)
        # patch = self.colorjitter(patch)
        if random.uniform(0, 1) < 0.5:
            patch = torch.flip(patch, dims=(-1,))
            patch_mask = torch.flip(patch_mask, dims=(-1,))
        return patch, patch_mask


class MaskRWPatchDataset(Dataset):
    def __init__(self, root, psize=512) -> None:
        super().__init__()
        self.ori_imgs = sorted(glob(osp.join(root, "ori", "*.jp*g")))
        self.phy_imgs = sorted(glob(osp.join(root, "phy", "*.jp*g")))
        self.masks = sorted(glob(osp.join(root, "mask", "*.jp*g")))
        assert len(self.ori_imgs) == len(self.masks) and len(self.ori_imgs) == len(self.phy_imgs)
        self.totensor = tv.transforms.Compose([
            tv.transforms.Resize((psize, psize)),
            tv.transforms.ToTensor(),
        ])
        self.psize = psize

    def __len__(self):
        return len(self.ori_imgs)

    def __getitem__(self, index):
        ori_img = Image.open(self.ori_imgs[index]).convert("RGB")
        phy_img = Image.open(self.phy_imgs[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("RGB")
        aug_param = tv.transforms.RandomResizedCrop.get_params(ori_img, (0.25, 1.0), (0.75, 1.33))
        ori_img = tv.transforms.functional.crop(ori_img, *aug_param)
        phy_img = tv.transforms.functional.crop(phy_img, *aug_param)
        mask = tv.transforms.functional.crop(mask, *aug_param)
        ori_img = self.totensor(ori_img)
        phy_img = self.totensor(phy_img)
        mask = self.totensor(mask)
        if random.uniform(0, 1) < 0.5:
            ori_img = torch.flip(ori_img, dims=(-1,))
            phy_img = torch.flip(phy_img, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))
        return ori_img, phy_img, mask


def coco_val_full_loader(psize=512, batch_size=16, num_workers=8, split="val", aug=lambda x:x):
    dataset = FullPatchDatasetV2(f"../datasets/COCO/{split}2017", psize=psize, aug=aug)
    length = len(dataset)
    trainset, testset = random_split(
        dataset,
        [int(0.8 * length), length - int(0.8 * length)],
        generator=torch.Generator().manual_seed(1119),
    )
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def coco_val_mask_loader(psize=512, batch_size=16, num_workers=8):
    dataset = MaskPatchDatasetV2(
        "../datasets/COCO/val2017", 
        "../datasets/COCO/val2017_mask", 
        psize=psize,
    )
    length = len(dataset)
    trainset, testset = random_split(
        dataset,
        [int(0.8 * length), length - int(0.8 * length)],
        generator=torch.Generator().manual_seed(1119),
    )
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader
