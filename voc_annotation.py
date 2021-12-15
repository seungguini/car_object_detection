import os
import random
import xml.etree.ElementTree as ET

from utils.utils import get_classes

# for HPC users, please uncomment the following line and change directory to project working directory
# os.chdir("/content/drive/MyDrive/YOLO_3")

# -------------------------------------------------------------------#
# path to the classes our dataset contains
# this is the same classes path as the path we used for training and predicting
# -------------------------------------------------------------------#
classes_path = 'model_data/cityscape_classes.txt'

# path to annotation files we converted using json_to_xml.py and tranform_cityscape.py
annotation_path = 'resource/gtFine'

train_val = ["train", "val"]
classes, _ = get_classes(classes_path)

# helper function that reads the object annotation from a single annotation file and write file_path & labels into cityscape_train.txt and cityscape_val.txt for training
def convert_annotation(kind, city, file_name, txt_file):
    in_file = open(os.path.join(os.path.abspath("resource"), 'gtFine/gtFine/%s/%s/%s.xml' % (kind, city, file_name)), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        txt_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


if __name__ == "__main__":
    random.seed(0)

    for kind in train_val:
        txt_file = open("cityscape_%s.txt" % kind, "w+", encoding="utf-8")

        for city in os.listdir(os.path.join(annotation_path, kind)):
            for file_name in os.listdir(os.path.join(annotation_path, kind, city)):
                if file_name[-3:] == "xml":
                    txt_file.write(os.path.join('%s/leftImg8bit/%s/%s/%s_leftImg8bit.png' % (
                        os.path.abspath("resource"), kind, city, file_name[:-24])))
                    convert_annotation(kind, city, file_name[:-4], txt_file)
                    txt_file.write("\n")
                    # for logging and tracking purposes
                    print("Generated path for " + os.path.join('%s/leftImg8bit/%s/%s/%s.png' % (
                        os.path.abspath("resource"), kind, city, file_name)))
        txt_file.close()

    print("Generate cityscape_train.txt and cityscape_val.txt for train done.")
