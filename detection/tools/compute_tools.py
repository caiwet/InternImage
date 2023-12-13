import json
import numpy as np
import math
from tabulate import tabulate

def compute_distance(bbox1, bbox2):
    midx1 = bbox1[0] + bbox1[2]
    midy1 = bbox1[1] + bbox1[3]
    midx2 = bbox2[0] + bbox2[2]
    midy2 = bbox2[1] + bbox2[3]
    return math.dist([midx1, midy1], [midx2, midy2])

def get_dis(gt, max_score, cls_id):
    dis = []
    img = set()
    test_ids = [i['id'] for i in gt['images']]
    large_err = []

    for test_id in test_ids:
        if test_id == 4146631:
            continue
        if (test_id, cls_id) not in max_score.keys():
            continue
        for ann in gt['annotations']:
            if ann['image_id'] == test_id and ann['category_id'] == cls_id:
                if test_id == 2946054 and ann['id'] == 5689277: ## manually skip wrong label
                    continue
                if test_id in img:
                    # print(ann)
                    continue
                gt_bbox = ann['bbox']
                pd_bbox = max_score[(test_id, cls_id)][0]['bbox']
                d = compute_distance(gt_bbox, pd_bbox)
                if d > 64:
                    # print(test_id)
                    large_err.append(test_id)
                dis.append(d)
                img.add(test_id)

    return dis, large_err
def get_dis_metrics(carina_dis, tip_dis, clavicles=False):
    first = 'carina'
    second = 'tip'
    if clavicles:
        first = 'left clavicle'
        second = 'right clavicle'
    print(f"Max {first} distance: ", np.max(carina_dis))
    print(f"Min {first} distance: ", np.min(carina_dis))
    print(f"Mean {first} distance: ", np.mean(carina_dis))
    print(f"Std {first} distance: ", np.std(carina_dis))
    print(f"Median {first} distance: ", np.median(carina_dis))

    print(f"Max {second} distance: ", np.max(tip_dis))
    print(f"Min {second} distance: ", np.min(tip_dis))
    print(f"Mean {second} distance: ", np.mean(tip_dis))
    print(f"Std {second} distance: ", np.std(tip_dis))
    print(f"Median {second} distance: ", np.median(tip_dis))

def get_max_pred_bbox_3cls(pred_file='mask_rcnn_test-dev_results.bbox.json'):
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
            elif pred['category_id'] == 3048:
                curr = max_score[key]
                if len(curr) < 2:
                    curr.append(pred)
                elif pred['score'] > min(curr[0]['score'], curr[1]['score']):
                        if curr[0]['score'] < curr[1]['score']:
                            curr[0] = pred
                        else:
                            curr[1] = pred
    return max_score

def get_max_pred_bbox_4cls(pred_file='mask_rcnn_test-dev_results.bbox.json'):
    with open(pred_file) as f:
        data = json.load(f)
        max_score = {}
        for pred in data:
            key = (pred['image_id'], pred['category_id'])
            if key not in max_score.keys():
                max_score[key] = [pred]
            if pred['score'] > max_score[key][0]['score']:
                max_score[key] = [pred]
    return max_score

def get_no_pred_id(max_score, cls_id=3047):
    pred_id = {}
    for (image_id, cat_id) in max_score.keys():
        if image_id not in pred_id.keys():
            pred_id[image_id] = []
        pred_id[image_id].append(cat_id)
    ids = []
    for key in pred_id.keys():
        if cls_id not in pred_id[key]:
            ids.append(key)
    return ids

# Return a map -> if no label in gt, then true; else false
def get_no_gt_id(gt, cls_id=3047):
    test_ids = [i['id'] for i in gt['images']]
    gt_no_label = {}
    for test_id in test_ids:
        gt_no_label[test_id] = True
        for ann in gt['annotations']:
            if ann['image_id'] == test_id and ann['category_id'] == cls_id:
                gt_no_label[test_id] = False
                continue
    return gt_no_label

def get_f1(max_score, gt, cls_id=3047):
    no_pred_ids = get_no_pred_id(max_score, cls_id=cls_id)
    no_gt_ids = get_no_gt_id(gt, cls_id=cls_id)
    tp, tn, fp, fn = 0, 0, 0, 0
    for id in no_pred_ids:
        if no_gt_ids[id] == True:
            tn += 1
        else:
            fn += 1
    actual_p = sum(val==False for val in no_gt_ids.values())
    actual_n = len(no_gt_ids.keys()) - actual_p
    tp = actual_p - fn
    fp = actual_n - tn

    precision = tp/(tp + fp)
    recall = tp/(tp+fn)
    f1 = 2*precision*recall/(precision+recall)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 score: ", f1)
    mydata = [
        ["Tip", tp, fp],
        ["No Tip", fn, tn]
    ]

    # create header
    head = ["", "Tip", "No Tip"]

    # display table
    print(tabulate(mydata, headers=head, tablefmt="grid"))
