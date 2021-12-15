# Car Object Detection
```bash
Repository for the CSCI-UA 9473 final project, 2021 FALL semester. 
Object detection specialized in driving situations.
```
Special Credit to user ***@Bubbliiiing*** on GitHub

***
## YOLO 3 Algorithm Introduction

YOLO stands for ***You Only Look Once***. Unlike traditional object detection algorithms, which use different neural networks for generating bounding boxes and classification, YOLO uses a single neural network and is trained to do classification and bounding box regression at the same time. Because YOLO only uses one neural network, in practical it runs a lot faster than ***faster rcnn*** and can converge quicker when training.

## How to train

***
**For details, please refer to the documentaion of the files mentioned below**

0) Create virutal environment and install Python dependencies:
```bash
# Create Python virtual environment
python3 -m venv venv

# Activte virtual environment
source venv/bin/activate

# Install dependencies
python3 install -r requirements.txt
```
1) Download the `gtFine_trainvaltest` dataset and place at the root of this repository
2) Convert Cityscape `.json` files to YOLO3-specific `.json` files with
```bash
python3 scripts/transform_cityscape.py
```
3) Convert `.json` files to `.xml` files with
```bash
python3 scripts/json_to_xml.py
```
4) Resize CityScape images into 416x416 images with padding on both top and bottom
```bash
python3 scripts/image_resize.py
```
5) Run voc_annotation.py to create `cityscape_train.txt` and `cityscape_val.txt` with
```bash
python3 voc_annotation.py
```
6) Run `train.py` to start train (specifications on train settings, see in `train.py` documentation)
```bash
python3 train.py
```
## 