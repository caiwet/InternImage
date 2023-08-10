import json
import numpy as np
import pandas as pd
import math
from tabulate import tabulate
import os

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

def generate_hospital_labels():
    root = "/n/data1/hms/dbmi/rajpurkar/lab/ett/hospital_downsized"

    for hospital in os.listdir(root):
        if hospital not in ["images", "annotations"]:
            get_gt_label(gt_annotation=os.path.join(root, hospital, "annotations/annotations.json"),
                         out_file=f"labels/gt_labels_{hospital}.csv")

def get_max_pred_bbox(pred_file='test_gloria.bbox.json'):
    with open(pred_file) as f:
        data = json.load(f)
        max_score = {}
        for pred in data:
            key = (pred['image_id'], pred['category_id'])
            if key not in max_score.keys():
                max_score[key] = [pred]
            elif pred['category_id'] == 3046 or pred['category_id'] == 3047:
                if pred['score'] > max_score[key][0]['score']:
                    max_score[key] = [pred]
            # elif pred['category_id'] == 3048:
            #     curr = max_score[key]
            #     if len(curr) < 2:
            #         curr.append(pred)
            #     elif pred['score'] > min(curr[0]['score'], curr[1]['score']):
            #             if curr[0]['score'] < curr[1]['score']:
            #                 curr[0] = pred
            #             else:
            #                 curr[1] = pred
    return max_score

def get_labels(max_score, thres=0):
    pred_labels = []
    for item in max_score:
        item=item[0]
        if item['score'] > thres:
            x = item['bbox'][0] + item['bbox'][2]/2
            y = item['bbox'][1] + item['bbox'][3]/2
            pred_labels.append([item['image_id'], item['category_id'], x, y, item['score']])
    pred_labels = pd.DataFrame(data=pred_labels, columns=["image_id", "category_id", "x", "y", "prob"])
    return pred_labels

def get_pred_labels(pred_file, outfile):
    max_score = get_max_pred_bbox(pred_file)
    max_score = list(max_score.values())
    pred_mimic = get_labels(max_score, thres=0)
    pred_mimic.to_csv(outfile, index=False)



if __name__ == "__main__":
    # Get gt label
    get_gt_label(
        gt_annotation="/n/data1/hms/dbmi/rajpurkar/lab/ett/hospital_downsized/Chiang_Mai_University/split/annotations/val_annotations.json",
        out_file=f"labels/gt_labels_Chiang_Mai_University_val.csv")

    # Get hospital gt label in a bundle
    # generate_hospital_labels()

    # Get pred label
    # get_pred_labels('gloria_NYU_Langone_Health.bbox.json', "labels/hospitals/NYU_Langone_Health_pred_labels.csv")
    # get_pred_labels('metric_v2.bbox.json', "labels/metric_v2_mimic.csv")


