import math
from tabulate import tabulate
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class ETTEvaler:
    """
    gt_labels: df with groundtruth information in the format of image_id, category_id, x, y
    pred_labels: df with predicted information in the format of image_id, category_id, x, y
    resized_dim: resized dimension of images, integer
    """
    def __init__(self, gt_labels, pred_labels, resized_dim) -> None:
        assert isinstance(resized_dim, int), "resized_dim should be an int"

        self.encode = {'tip': 1,
                       'carina': 0}
        self.gt_labels = gt_labels
        self.pred_labels = pred_labels
        self.thres = resized_dim
        self.resized_dim = resized_dim
        self.convert = pd.read_csv("/home/cat302/ETT-Project/ETT_Evaluation/pixel_spacing.csv")

    """
    Get metrics report for error between prediction and groundtruth.

    Parameters:
        tp: number of true positive
        fp: number of false positive
        tn: number of true negative
        fn: number of false negative
        distances: list of distances between prediction and groundtruth
        cat: 'tip' or 'carina'
    """
    def get_metrics(self, tp, fp, tn, fn, distances, cat='tip'):
        print("**************************************")
        print(f"Max {cat} distance: ", np.max(distances))
        print(f"Min {cat} distance: ", np.min(distances))
        print(f"Mean {cat} distance: ", np.mean(distances))
        print(f"Std {cat} distance: ", np.std(distances))
        print(f"Median {cat} distance: ", np.median(distances))

        print("**************************************")
        precision = tp/(tp + fp)
        recall = tp/(tp+fn)
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        f1 = 2*precision*recall/(precision+recall)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("Sensitivity: ", sensitivity)
        print("Specificity: ", specificity)
        print("F1 score: ", f1)
        mydata = [
            [cat, tp, fp],
            [f"No {cat}", fn, tn]
        ]

        # create header
        head = ["", cat, f"No {cat}"]

        # display table
        print(tabulate(mydata, headers=head, tablefmt="grid"))

    """
    Compute euclidean distance between actual point and predicted point

    Parameters:
        target: tip or carina
    """
    def gt_pred_distance(self, target='tip'):

        tp, fp, tn, fn = 0, 0, 0, 0
        distances = []

        for img_id in self.pred_labels['image_id'].unique():
            gt_label = self.gt_labels[(self.gt_labels['image_id']==img_id) & (self.gt_labels['category_id']==self.encode[target])]
            pred_label = self.pred_labels[(self.pred_labels['image_id']==img_id) & (self.pred_labels['category_id']==self.encode[target])]
            target_exist = True if len(gt_label)>0 else False
            target_predicted = True if len(pred_label)>0 else False
            if len(gt_label) > 1:
                raise ValueError("Repeated label in groundtruth file.")
            if len(pred_label) > 1:
                raise ValueError("Repeated label in prediction file.")
            if not target_exist:
                if not target_predicted:
                    tn += 1
                else:
                    fp += 1
            elif not target_predicted:
                fn += 1
            else:
                tp += 1
                # dis = math.dist((gt_label['x'], gt_label['y']), (pred_label['x'], pred_label['y']))
                dis = ((float(gt_label['x'])-float(pred_label['x']))**2 + (float(gt_label['y'])-float(pred_label['y']))**2)**0.5
                # breakpoint()
                if dis > self.thres: # random threshold for abnormal data
                    print(f"Possibly wrong image {img_id} with distance {dis}")
                    continue
                dis = self.pixel_to_cm(img_id, dis)
                if dis:
                    distances.append(dis)
        return tp, fp, tn, fn, distances

    def pixel_to_cm(self, img_id, dis):
        tmp = self.convert[self.convert['image_id']==img_id]
        if len(tmp) == 0:
            return None
        cropped_size = min(int(tmp['original_width']), int(tmp['original_height']))
        ps = cropped_size * float(tmp['pixel_spacing_x']) / self.resized_dim  ## Assume x and y have the same pixel spacing

        return ps * dis * 0.1


    def __tip_carina_distance(self, labels):
        id_to_distances = {}
        for image_id in labels['image_id'].unique():
            label = labels[labels['image_id'] == image_id]

            x_tip = label[label['category_id']==self.encode['tip']]['x']
            y_tip = label[label['category_id']==self.encode['tip']]['y']
            x_carina = label[label['category_id']==self.encode['carina']]['x']
            y_carina = label[label['category_id']==self.encode['carina']]['y']

            if len(x_tip)<1 or len(x_tip)<1:
                # print("Tip not exist for image ", image_id)
                continue
            if len(x_carina)<1 or len(x_carina)<1:
                # print("Carina not exist for image ", image_id)
                continue
            if len(x_carina) > 1:
                # print("Multiple label exists")
                # print(image_id)
                # print(label[label['category_id']==self.encode['carina']])
                continue

            if len(x_tip) > 1:
                # print("Multiple label exists")
                # print(image_id)
                # print(label[label['category_id']==self.encode['tip']])
                continue


            dis = math.dist((float(x_tip), float(y_tip)), (float(x_carina), float(y_carina)))
            if dis > self.thres:
                print(f"Possibly wrong image {image_id} with distance {dis}")
                continue
            dis = self.pixel_to_cm(image_id, dis)
            if dis:
                id_to_distances[image_id] = dis
        return id_to_distances

    """
    Compute distance between tip and carina. If gt is True, then use the groundtruth
    label, otherwise use predicted label.
    """
    def tip_carina_distance(self, gt=True):
        if gt:
            print("Processing groundtruth label...")
            id_to_distances = self.__tip_carina_distance(self.gt_labels)
        else:
            print("Processing predicted label...")
            id_to_distances = self.__tip_carina_distance(self.pred_labels)
        return id_to_distances


    def plot_dis(self, distances, unit="pixel", cat="Tips"):
        plt.hist(distances)
        plt.xlabel(f"Pixel Distance in {unit}")
        plt.ylabel("Frequency")
        plt.title(f"Distance between Predicted Point and Actual Point for {cat}")

    """
    Calculate the mean squared error between the groundtruth tip-carina distance
    and predicted tip-carina distance.
    """
    def mean_squared_error(self, dict1, dict2):
        shared_keys = set(dict1.keys()) & set(dict2.keys())
        if not shared_keys:
            raise ValueError("Dictionaries have no shared keys.")
        mse = {}
        for key in shared_keys:
            mse[key] = (dict1[key] - dict2[key]) ** 2
        return sum(mse.values()) / len(mse)

    """
    Calculate the r-squared between the groundtruth tip-carina distance
    and predicted tip-carina distance.
    """
    def calculate_r_squared(self, dict1, dict2):
        shared_keys = set(dict1.keys()) & set(dict2.keys())
        if not shared_keys:
            raise ValueError("Dictionaries have no shared keys.")
        actual = [dict1[key] for key in shared_keys]
        predicted = [dict2[key] for key in shared_keys]
        residuals = [actual[i] - predicted[i] for i in range(len(actual))]
        ss_res = sum([residuals[i] ** 2 for i in range(len(residuals))])
        mean_actual = sum(actual) / len(actual)
        ss_tot = sum([(actual[i] - mean_actual) ** 2 for i in range(len(actual))])
        r_squared = 1 - (ss_res / ss_tot)
        return r_squared