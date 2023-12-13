# train: 2360
# val: 507
import os
import json
import pandas as pd
import numpy as np
import random
import shutil
from PIL import Image   


def split_images(input_folder, output_folder):
    train_csv = pd.read_csv("/n/data1/hms/dbmi/rajpurkar/lab/MAIDA_ETT/RANZCR/RANZCR-all-train.csv")
    val_csv = pd.read_csv("/n/data1/hms/dbmi/rajpurkar/lab/MAIDA_ETT/RANZCR/RANZCR-all-val.csv")
    test_csv = pd.read_csv("/n/data1/hms/dbmi/rajpurkar/lab/MAIDA_ETT/RANZCR/RANZCR-all-test.csv")
    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
    
    train_files = []
    val_files = []
    test_files = []
    # Split the images and move them to the respective output folders
    for image_file in image_files:
        source_path = os.path.join(input_folder, image_file)
        image_name = image_file[:-4]
        if image_name in list(train_csv['StudyInstanceUID']):
            destination_path = os.path.join(output_folder, 'train', image_file)
            train_files.append(destination_path)
        elif image_name in list(val_csv['StudyInstanceUID']):
            destination_path = os.path.join(output_folder, 'val', image_file)
            val_files.append(destination_path)
        elif image_name in list(test_csv['StudyInstanceUID']):
            destination_path = os.path.join(output_folder, 'test', image_file)
            test_files.append(destination_path)
        else:
            print(image_file)
        # shutil.copy(source_path, destination_path)
    return train_files, val_files, test_files

def create_anno(image_files, anno_path, id_prefix):
    with open(anno_path) as f:
        anno = json.load(f)
    for i, image_file in enumerate(image_files):
    #     image_id = id_prefix + i
    #     anno['images'].append({'id': image_id, 'coco_url': '', 'width': 1280, 'height': 1280, 'date_captured': '',
    #                            'file_name': image_file})
    # anno['categories'] = [{'id': 3043, 'name': 'Invalid', 'supercategory': ''},
    #                       {'id': 3042, 'name': 'No Finding', 'supercategory': ''},
    #                       {'id': 1, 'name': 'tip', 'supercategory': ''}]
        for an in anno['images']:
            if an['id'] == id_prefix+i and 'file_name' not in an.keys():
                an['file_name'] = image_file

    with open(anno_path, 'w') as f:
        json.dump(anno, f)


image_path = '/n/data1/hms/dbmi/rajpurkar/lab/MAIDA_ETT/ranzcr_no_ett_all_data/ranzcr_no_ett'
ratio = 2360 / (2360+507)
train_anno = '/n/data1/hms/dbmi/rajpurkar/lab/MAIDA_ETT/ranzcr_no_ett_all_data/annotations/train_annotations_enl5.json'
val_anno = '/n/data1/hms/dbmi/rajpurkar/lab/MAIDA_ETT/ranzcr_no_ett_all_data/annotations/val_annotations_enl5.json'

train_files, val_files, test_files = split_images(input_folder='/n/data1/hms/dbmi/rajpurkar/lab/MAIDA_ETT/ranzcr_no_ett_all_data/ranzcr_no_ett', 
             output_folder='/n/data1/hms/dbmi/rajpurkar/lab/MAIDA_ETT/ranzcr_no_ett_all_data/images') 

create_anno(image_files=train_files,
            anno_path=train_anno, id_prefix=1000)
create_anno(image_files=val_files,
            anno_path=val_anno, id_prefix=2000)