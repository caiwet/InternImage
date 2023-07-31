import json
import numpy as np
import pandas as pd
import math
from tabulate import tabulate

def get_gt_label(gt_annotation, out_file):   

    with open(gt_annotation) as f:
        gt_data = json.load(f)

    id_to_filename = {}
    for img in gt_data["images"]:
        id_to_filename[img['id']] = img['file_name'][:-4]

    gt_labels = []
    for anno in gt_data['annotations']:
        if anno['image_id'] in id_to_filename.keys():
            file = id_to_filename[anno['image_id']]
            bbox = anno['bbox']
            x = bbox[0] + bbox[2] / 2
            y = bbox[1] + bbox[3] / 2
            gt_labels.append([anno['image_id'], anno['category_id'], x, y])
    gt_df = pd.DataFrame(gt_labels, columns=['image_id', 'category_id', 'x', 'y'])
    # display(gt_df)
    # print(len(gt_df["image_id"].unique()))
    gt_df.to_csv(out_file, index=False)  



if __name__ == "__main__":
#     get_gt_label(gt_annotation="/n/data1/hms/dbmi/rajpurkar/lab/ett/all_data_split/annotations/val_annotations_enl5.json",
#                  out_file="labels/gt_labels_all_data_val.csv")
#     get_gt_label(gt_annotation="/n/data1/hms/dbmi/rajpurkar/lab/ett/Test/downsized/MIMIC/annotations/test_annotations_enl5.json",
#                  out_file="labels/gt_labels_mimic.csv")
#     get_gt_label(gt_annotation="/n/data1/hms/dbmi/rajpurkar/lab/ett/Test/downsized/RANZCR/annotations/test_annotations_enl5.json",
#                  out_file="labels/gt_labels_ranzcr.csv")
    institutions = ['Austral', 'Cedars-Sinai', 'Chiang_Mai_University', 'Morales_Meseguer_Hospital', 'Newark_Beth_Israel_Medical_Center', 'NYU_Langone_Health', 'Osaka_City_University', 'Technical_University_of_Munich', 'Universitätsklinikum_Tübingen', 'University_of_Miami']
    for ins in institutions:
        get_gt_label(gt_annotation=f"/n/data1/hms/dbmi/rajpurkar/lab/ett/hospital_downsized/{ins}/annotations/annotations.json",
                     out_file=f"labels/gt_labels_{ins}.csv")