# ***CamPro***: Camera-based Anti-Facial Recognition

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10156141.svg)](https://doi.org/10.5281/zenodo.10156141)

> Accepted by Network and Distributed System Security (NDSS) Symposium 2024

## Requirements

### Hardware

- more than 64 GB disk space
- at least one GPU that supports CUDA (Recommended: NVIDIA RTX 3090)

### Software

- a recent Linux OS (e.g., Ubuntu 18.04/20.04/22.04)
- Anaconda
- CUDA driver whose version should be newer than 11.3

## Semi-Auto Setup (Recommended)

- First, download all the three folders from the [Google Drive](https://drive.google.com/drive/folders/1fvXBKqukA2BnGQU76QLtsRShuA5eiLY7?usp=sharing) and place them under this directory
- Second, run the script `start.sh` in your terminal. It will process the datasets and create the python virtual environment named `CamPro`.

## Manually Setup

### Python Environment Setup

We assume that you have installed the Python 3.9, and the CUDA Version should be higher than 11.3 (You can check this via `nvidia-smi`).

- If you have installed [`Anaconda`](https://www.anaconda.com/download), you could create a new environment via the command `conda create --yes --name CamPro python=3.9`.
- Then, you can activate the environment via the command `conda activate CamPro`.
- Finally, you should install the required packages via `pip install -r requirements.txt`.

### Data Preparation

> Google Drive: https://drive.google.com/drive/folders/1fvXBKqukA2BnGQU76QLtsRShuA5eiLY7?usp=sharing

- Datasets
    - CelebA: You should download the `CelebA.zip` from the Google Drive and unzip it under the `datasets` folder.
    - LFW: You should download the `LFW.zip` from the Google Drive and unzip it under the `datasets` folder.
    - COCO: 
        - You should download the `COCO.zip` from the Google Drive and unzip it under the `datasets` folder. Then, make two empty folders, i.e., `images` and `val2017`, under the path `datasets/COCO`.
        - You should download `COCO Detection 2017` images from the official website:
            - http://images.cocodataset.org/zips/train2017.zip
            - http://images.cocodataset.org/zips/val2017.zip
        - Extract all the JPEG images in `train2017.zip` to `datasets/COCO/images`.
        - Extract all the JPEG images in `val2017.zip` to `datasets/COCO/val2017` and `datasets/COCO/images`.
- Models
    - checkpoints: You should download the `checkpoints` folder from the Google Drive.
    - weights: You should download the `weights` folder from the Google Drive.


## Directories

After finishing data preparation, the `CamPro` directory should be organized like this below.

```
CamPro
├── checkpoints         # trained weights of various models
├── datasets            # open image datasets
│   ├── CelebA          # 112x112 cropped face dataset
│   ├── COCO            # person detection dataset
│   │   ├── images
│   │   ├── labels
│   │   ├── val2017
│   │   ├── val2017_mask
│   │   ├── test.txt
│   │   └── train.txt
│   └── LFW             # 112x96 cropped face dataset
├── results             # !!! collected results for artifact evaluation !!!
├── src                 # source codes
│   ├── baseline        # baseline methods of privacy protection
│   ├── evaluation      # !!! main python scripts for artifact evaluation !!!
│   ├── isp             # functions/classes related to ISP
│   ├── misc            # some helper functions
│   ├── privacy         # functions/classes related to privacy evaluation
│   ├── unet            # functions/classes related to image enhancer
│   ├── utility         # functions/classes related to utility evaluation
│   └── yolov5          # open-sourced object detection code repository
└── weights             # pre-trained weights of various models
```

## Artifact Evaluation

In the following, we briefly introduce the execution steps. The details can be found in the AE appendix.

### Anti-Facial Recognition Experiment (E1)

- conda activate CamPro
- cd src/
- python evaluation/exp1.py
- **Results are saved to `results/1.csv`.**

### Ablation Study Experiment (E2)

- conda activate CamPro
- cd src/
- python evaluation/exp2.py
- **Results are saved to `results/2.csv`.**

### Vision Application Performance Experiment (E3)

- conda activate CamPro
- cd src/
- python evaluation/exp3.py
- **Results are saved to `results/3.csv`.**

### Image Quality Assessment Experiment (E4)

- conda activate CamPro
- cd src/
- python evaluation/exp4.py
- **Results are saved to `results/4.csv`.**

### White-box Adaptive Attack Experiment (E5)

- conda activate CamPro
- cd src/
- python evaluation/exp5.py
- **Results are printed in the console.**
