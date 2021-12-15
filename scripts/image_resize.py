# !/user/bin/env python    
# -*- coding:utf-8 -*-
import os
import cv2

# for HPC users, please uncomment the following line and change directory to project working directory
# os.chdir("/content/drive/MyDrive/YOLO_3")

# Convert the images to 416x416, also padding it with black on top and bottom
img_root_path = "/content/drive/MyDrive/YOLO_3/resource/leftImg8bit"

img_folder_list = ["test", "train", "val"]

for test_train_val in img_folder_list:
	for city in os.listdir(img_root_path + "/" + test_train_val):
		for file_name in os.listdir(img_root_path + "/" + test_train_val + "/" + city):
			img_temp = cv2.imread(
				img_root_path + "/" + test_train_val + "/" + city + "/" + file_name)
			img_resize = cv2.resize(img_temp, (416, 208))
			img_resize = cv2.copyMakeBorder(img_resize, 104, 104, 0, 0, borderType=cv2.BORDER_CONSTANT)
			cv2.imwrite(img_root_path + "/" + test_train_val + "/" + city + "/" + file_name, img_resize)
			print(
				"Resized " + img_root_path + "/" + test_train_val + "/" + city + "/" + file_name + " from " + str(img_temp.shape)
			)
