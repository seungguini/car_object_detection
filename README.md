# car_object_detection
Repository for the Introduction to Machine Learning's final project. Object detection for cars, pedestrians, and things on the road.

## Convert Cityscape `.json` file to `.xml` file

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
2) Convert Cityscape `.json` files to YOLO-specific `.json` files with
```bash
python3 scripts/transform_cityscape.py
```
3) Convert `.json` files to `.xml` files
- _Code work in progress_