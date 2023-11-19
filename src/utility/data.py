import yaml
import os.path as osp
from torch.utils.data import DataLoader
from utils.datasets import LoadImagesAndLabels

dataset_folder = osp.join(
    osp.dirname((osp.abspath(__file__))),
    "..",
    "..",
    "datasets",
    "COCO",
)


train_list = osp.join(dataset_folder, "train.txt")
test_list = osp.join(dataset_folder, "test.txt")

yolov5_root = osp.join(osp.dirname(osp.abspath(__file__)), "..", "yolov5")
hyp_yaml = osp.join(yolov5_root, "data", "hyps", "hyp.scratch-low.yaml")
with open(hyp_yaml, "r") as f:
    hyp = yaml.safe_load(f)


def coco_person_dataloader(batch_size, num_workers):
    trainset = LoadImagesAndLabels(train_list, augment=True, hyp=hyp)
    testset = LoadImagesAndLabels(test_list, augment=False, hyp=hyp)
    loader1 = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=LoadImagesAndLabels.collate_fn,
    )
    loader2 = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=LoadImagesAndLabels.collate_fn,
    )
    return loader1, loader2
