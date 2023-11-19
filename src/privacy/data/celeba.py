import torch
import torchvision as tv
from torch.utils.data import Dataset, DataLoader, random_split
import os.path as osp
import random
from glob import glob
from PIL import Image
from collections import Counter

dataset_folder = osp.join(
    osp.dirname((osp.abspath(__file__))),
    "..",
    "..",
    "..",
    "datasets",
    "CelebA",
)


def celeba_dataloader(batch_size, num_workers):
    augment_transform = tv.transforms.Compose(
        [
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
        ]
    )
    dataset = tv.datasets.ImageFolder(dataset_folder, augment_transform)
    length = len(dataset)
    trainset, testset = random_split(
        dataset,
        [int(0.8 * length), length - int(0.8 * length)],
        generator=torch.Generator().manual_seed(100),
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


class ClosedSetCelebA(Dataset):
    def __init__(self) -> None:
        super().__init__()
        root = osp.join(dataset_folder, "**", "*.png")
        imgs = glob(root)
        random.shuffle(imgs)
        # filter num image larger than 1
        cnt = Counter([x.split("/")[-2] for x in imgs])
        pid = 0
        valid = {}
        for k, v in cnt.items():
            if v > 1:
                valid[k] = pid
                pid += 1
        # sample gallery and probe images
        self.gallery = [None] * len(valid)
        self.probe = [None] * len(valid)
        for img in imgs:
            k = img.split("/")[-2]
            if k in valid:
                i = valid[k]
                if self.gallery[i] is None:
                    self.gallery[i] = img
                elif self.probe[i] is None:
                    self.probe[i] = img
        self.transform = tv.transforms.ToTensor()

    def __len__(self):
        return len(self.gallery)

    def __getitem__(self, index):
        gallery = self.transform(Image.open(self.gallery[index]))
        probe = self.transform(Image.open(self.probe[index]))
        return gallery, probe


class ClosedSetCelebA2(Dataset):
    def __init__(self, imgs) -> None:
        super().__init__()
        random.shuffle(imgs)
        # filter num image larger than 1
        cnt = Counter([x.split("/")[-2] for x in imgs])
        pid = 0
        valid = {}
        for k, v in cnt.items():
            if v > 1:
                valid[k] = pid
                pid += 1
        # sample gallery and probe images
        self.gallery = [None] * len(valid)
        self.probe = [None] * len(valid)
        for img in imgs:
            k = img.split("/")[-2]
            if k in valid:
                i = valid[k]
                if self.gallery[i] is None:
                    self.gallery[i] = img
                elif self.probe[i] is None:
                    self.probe[i] = img
        self.transform = tv.transforms.ToTensor()

    def __len__(self):
        return len(self.gallery)

    def __getitem__(self, index):
        gallery = self.transform(Image.open(self.gallery[index]))
        probe = self.transform(Image.open(self.probe[index]))
        return gallery, probe

