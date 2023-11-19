#!/bin/bash

# before run this script
# we assume that you have manually downloaded the contents from the Google Drive sharing link
# Google Drive: https://drive.google.com/drive/folders/1fvXBKqukA2BnGQU76QLtsRShuA5eiLY7?usp=sharing

current_directory=$(pwd)

cd ./datasets
unzip CelebA.zip
unzip COCO.zip
unzip LFW.zip

# download coco dataset
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip train2017.zip
unzip val2017.zip

# prepare the data
cd ./COCO
mkdir images
mv ../val2017 ./
cp val2017/*.jpg images/
cd ../train2017
for num in 0 1 2 3 4 5 6 7 8 9
do
    mv *$num.jpg ../COCO/images/
done
cd ..
rm -r train2017

# create the runtime environment
cd $current_directory
conda create --yes --name CamPro python=3.9
conda activate CamPro
pip install -r requirements.txt
