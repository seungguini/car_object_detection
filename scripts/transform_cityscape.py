# Script to transform Cityscape data to Yolo v3 XML format
import pandas as pd
import os, json
import pandas as pd
import numpy as np

# Path to cityscape JSON files
CITYSCAPE_JSON = './cityscape_json/'


# 1. Grab all .JSON files
path_to_json = './gtFine_trainvaltest/gtFine/train/aachen/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

# 2. Create a JSON file

# Loop through each JSON file
for json_file in json_files:
    print(json_file)
    # Initialize json_text
    json_text = {}

    # Open file
    with open(path_to_json + json_file, 'r') as f:
        json_text = json.load(f)
    
    # Get all objects
    objects = json_text['objects']

    # Image size
    width = json_text['imgWidth']
    height = json_text['imgHeight']

    # Base JSON
    base_json = {
        "annotation": [
            {
                "folder": "cityscape"
            },
            {
                "filename": json_file
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
                        "depth": 0
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

        xmin = np.min(array[:,0])
        xmax = np.max(array[:,1])
        ymin = np.min(array[:,0])
        ymax = np.max(array[:,1])

        object_json = {
            "object": {
                'name': name,
                'bndbox' : [
                    {
                        'xmin': int(xmin)
                    },{
                        'ymin': int(ymin)
                    },{
                        'xmax': int(xmax)
                    },{
                        'ymax': int(ymax)
                    },
                ]
            }
        }
        # Store object JSON to 
        base_json['annotation'].append(object_json)

    with open(CITYSCAPE_JSON + json_file, 'w') as json_f:
        json.dump(base_json, json_f)
    break