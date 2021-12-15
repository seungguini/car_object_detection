import json
import os
import numpy as np
import re

# Script to transform Cityscape annotation to the JSON format that YOLO v3 can understand

# for HPC users, please uncomment the following line and change directory to project working directory
# os.chdir("/content/drive/MyDrive/YOLO_3")

#-------------------------------#
# Path to CityScape JSON files
gtFine_path = 'resource/gtFine'

# img downsize ratio
ratio = 2048 / 416

# go through all the .json annotation files in gtFine folder (CityScape dataset should be put under resource folder)
for test_train_val in os.listdir(gtFine_path):
    for place in os.listdir(gtFine_path + "/" + test_train_val):
        for file_name in os.listdir(gtFine_path + "/" + test_train_val + "/" + place):
            if re.findall("json$", file_name):

                if file_name[-8:-5] != "new":
                    print(
                        "Converting " + gtFine_path + "/" + test_train_val + "/" + place + "/" + file_name)

                    path_to_file = gtFine_path + "/" + test_train_val + "/" + place + "/" + file_name
                    # Init json text
                    json_text = {}

                    # Open file
                    with open(path_to_file, 'r') as f:
                        json_text = json.load(f)

                    # Get all objects
                    objects = json_text['objects']

                    # Image size
                    width = 416
                    height = 416

                    # Base JSON
                    base_json = {
                        "annotation": [
                            {
                                "folder": "cityscape"
                            },
                            {
                                "filename": file_name
                            },
                            {
                                "source": [
                                    {
                                        "database": "Cityscape"
                                    },
                                    {
                                        "annotation": "Cityscape"
                                    },
                                    {
                                        "image": "Cityscape"
                                    },
                                ]
                            },
                            {
                                "owner": [
                                    {
                                        "flickrid": "Cityscape"
                                    },
                                    {
                                        "name": "Cityscape"
                                    }
                                ]
                            },
                            {
                                "size": [
                                    {
                                        "width": width
                                    },
                                    {
                                        "height": height
                                    },
                                    {
                                        "depth": 3
                                    }
                                ]
                            },
                            {
                                "segmented": 0
                            },
                        ]

                    }

                    # Loop through each object
                    for object in objects:
                        name = object['label']
                        polygon = object['polygon']

                        array = np.array(polygon)

                        xmin = np.min(array[:, 0]) / ratio
                        xmax = np.max(array[:, 1]) / ratio
                        ymin = np.min(array[:, 0]) / ratio + 104
                        ymax = np.max(array[:, 1]) / ratio + 104

                        object_json = {
                            "object": {
                                'name': name,
                                'bndbox': [
                                    {
                                        'xmin': int(xmin)
                                    }, {
                                        'ymin': int(ymin)
                                    }, {
                                        'xmax': int(xmax)
                                    }, {
                                        'ymax': int(ymax)
                                    },
                                ]
                            }
                        }
                        # Store object JSON to
                        base_json['annotation'].append(object_json)

                    with open(path_to_file[:-5] + "_new.json", 'w+') as json_f:
                        json.dump(base_json, json_f)

