import os.path as osp
import random
import torchvision as tv
from torchvision.datasets import LFWPairs
from torch.utils.data import DataLoader, Dataset
from glob import glob
from PIL import Image
from collections import Counter


dataset_folder = osp.join(
    osp.dirname((osp.abspath(__file__))),
    "..",
    "..",
    "..",
    "datasets",
    "LFW",
)

class ClosedSetLFW(Dataset):
    def __init__(self) -> None:
        super().__init__()
        root = osp.join(dataset_folder, "**", "*.jpg")
        imgs = glob(root)
        random.shuffle(imgs)
        # filter num image larger than 1
        cnt = Counter([x.split('/')[-2] for x in imgs])
        pid = 0
        valid = {}
        for k, v in cnt.items():
            if v > 1:
                valid[k] = pid
                pid += 1
        # sample gallery and probe images
        self.gallery = [None] * len(valid)
        self.probe = [None] * len(valid)
        self.imposter = []
        for img in imgs:
            k = img.split('/')[-2]
            if k in valid:
                i = valid[k]
                if self.gallery[i] is None:
                    self.gallery[i] = img
                elif self.probe[i] is None:
                    self.probe[i] = img
            elif len(self.imposter) < len(valid):
                self.imposter.append(img)
        self.transform = tv.transforms.Compose([
            tv.transforms.Resize((112, 112)),
            tv.transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.gallery)
    
    def __getitem__(self, index):
        gallery = self.transform(Image.open(self.gallery[index]))
        probe = self.transform(Image.open(self.probe[index]))
        return gallery, probe

