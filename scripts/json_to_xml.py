from json2xml import json2xml
import json
import os
import re

# for HPC users, please uncomment the following line and change directory to project working directory
# os.chdir("/content/drive/MyDrive/YOLO_3")

# Path to CityScape JSON files
gtFine_path = 'resource/gtFine'

for test_train_val in os.listdir(gtFine_path):
  for place in os.listdir(gtFine_path + "/" + test_train_val):
    for file_name in os.listdir(gtFine_path + "/" + test_train_val + "/" + place):
      if re.findall("new\.json$", file_name):
        print("Converting " + gtFine_path + "/"+ test_train_val + "/" + place + "/" + file_name + " to xml")
        file_temp = json.load(open(gtFine_path + "/"+ test_train_val + "/" + place + "/" + file_name))
        xml_temp = json2xml.Json2xml(file_temp, item_wrap=False).to_xml()
        with open(gtFine_path + "/"+ test_train_val + "/" + place + "/" + file_name[:-5] + ".xml", "w+") as f:
          f.write(xml_temp)
