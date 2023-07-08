import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json

def plot_position_arrows(ids, file='/home/ec2-user/segmenter/MAIDA/data1000/test_annotations_4cls.json',
                         from_cls=3049, to_cls=3047):
    with open(file) as f:
        data = json.load(f)
    id_bbox_map = {}

    for test_id in ids:
        for ann in data['annotations']:
            if ann['image_id'] == test_id:
                if test_id not in id_bbox_map.keys():
                    id_bbox_map[test_id] = {}
                if ann['category_id'] not in id_bbox_map[test_id].keys():
                    id_bbox_map[test_id][ann['category_id']] = []
                id_bbox_map[test_id][ann['category_id']].append(ann['bbox'])
    vecs = []
    for id in id_bbox_map.keys():
        if from_cls not in id_bbox_map[id].keys():
            continue
        if to_cls not in id_bbox_map[id].keys():
            continue
        x1, y1, w1, h1 = id_bbox_map[id][from_cls][0]
        x2, y2, w2, h2 = id_bbox_map[id][to_cls][0]
        mid1 = np.array([x1+w1/2, y1+h1/2])
        mid2 = np.array([x2+w2/2, y2+h2/2])
        vecs.append(mid2-mid1)
    vecs = np.array(vecs)
    x = np.zeros(len(vecs))
    y = np.zeros(len(vecs))
    for cat in data['categories']:
        if cat['id'] == from_cls:
            from_name = cat['name']
        if cat['id'] == to_cls:
            to_name = cat['name']
    plt.title(f"Vector from {from_name} to {to_name}")
    plt.quiver(x, y, vecs[:, 0], vecs[:, 1], scale=150, scale_units='inches')
    plt.show()

def plot_dis(carina_dis, tip_dis, clavicles=False):
    first = 'Carina'
    second = 'Tip'
    if clavicles:
        first = 'Left Clavicle'
        second = 'Right Clavicle'

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].hist(carina_dis)
    axes[0].set_title(f"Distance for {first} Prediction")
    axes[0].set_xlabel("Distance in cm")
    axes[1].hist(tip_dis)
    axes[1].set_title(f"Distance for {second} Prediction")
    axes[1].set_xlabel("Distance in cm")