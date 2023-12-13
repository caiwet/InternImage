import numpy as np
import pandas as pd
import sklearn.metrics as metrics

class UpdatedMetric:
    def __init__(self, gt_labels, pred_labels, pixel_spacing_file, resized_dim, encode={"carina": 0, "tip": 1}):
        self.encode = encode
        self.pixel_spacing = pd.read_csv(pixel_spacing_file)
        self.pixel_spacing['image_id'] = self.pixel_spacing['image_id'].astype('int')
        self.gt_labels = gt_labels
        self.pred_labels = pred_labels
        self.resized_dim = resized_dim
        self.gt_carina, self.gt_ett = self.get_coords(file=self.gt_labels)
        self.pred_carina, self.pred_ett = self.get_coords(file=self.pred_labels)

    def get_coords(self, file, set_ratio=None):
        """Get coordinate of carina and ett
        Args:
            file: label csv for either training or testing set

        Outputs:
            carina: pandas dataframe of carina coordinates
            ett: pandas dataframe of ett coordinates
        """

        # Rescale based on pixel spacing
        # breakpoint()
        if set_ratio: # override ratio specified in the pixel_spacing_file
            file["x"] = file["x"] * set_ratio
            file["y"] = file["y"] * set_ratio
        else:
            for i in range(len(file)):
                img_id = file.loc[i, "image_id"]
                scale = self.pixel_to_cm(img_id)
                file.loc[i, "x"] = file.loc[i, "x"] * scale
                file.loc[i, "y"] = file.loc[i, "y"] * scale
        # breakpoint()
        # Filter for carina and ett
        carina = file[file["category_id"] == self.encode["carina"]]
        ett = file[file["category_id"] == self.encode["tip"]]
        return carina, ett


    def pixel_to_cm(self, img_id):
        """Convert distance in pixel to distance in cm

        Args:
            img_id: image id

        Returns:
            scale: ratio to multiply in order to convert pixel to cm
        """

        # return 0.03 # test for hospitals

        # Get data for image id
        self.pixel_spacing = self.pixel_spacing.astype({'image_id': 'int'})
        tmp = self.pixel_spacing[self.pixel_spacing['image_id']==img_id]
        if len(tmp) == 0:
            print(f"No pixel spacing: {img_id}")
            return 2500*0.139*0.1/self.resized_dim # a random default value
#             raise ValueError("Image id not found in pixel spacing file")

        # Crop based on min of width and height.
        cropped_size = min(int(tmp['original_width']), int(tmp['original_height']))
        ps = cropped_size * float(tmp['pixel_spacing_x']) / self.resized_dim # Assume x and y have the same pixel spacing
        scale = ps * 0.1 # multiply by 0.1 b/c pixel spacing conversion ratio in the csv file converts to mm
        return scale


    def filter_carina(self, gt, pred):
        """Filter for common image ids
        """
        # Get common image ids
        ids_intersection = set(gt["image_id"]).intersection(set(pred["image_id"]))
        ids_union = set(gt["image_id"]).union(set(pred["image_id"]))

        # Sanity check
        if len(ids_intersection) != len(ids_union):
            print("Warning: some images are missing in either gt or pred.")

        # Filter for intersecting image ids
        mask = gt["image_id"].isin(ids_intersection)
        gt = gt.loc[mask].sort_values(by=["image_id"])
        gt = gt[["x", "y"]].to_numpy()
        mask = pred["image_id"].isin(ids_intersection)
        pred = pred.loc[mask].sort_values(by=["image_id"])
        pred = pred[["x", "y"]].to_numpy()
        return gt, pred

    def filter_ett(self, gt_ett, pred_ett):
        # Get common image ids
        ids_intersection = set(gt_ett["image_id"]).intersection(set(pred_ett["image_id"]))
        ids_union = set(gt_ett["image_id"]).union(set(pred_ett["image_id"]))

        # Filter for intersecting image ids
        pred_binary, gt_binary, pred_prob, gt_coord, pred_coord = [], [], [], [], []
        gt_ids = gt_ett["image_id"].unique()
        pred_ids = pred_ett["image_id"].unique()
        for i in ids_union:
            # Get prediction
            if i in pred_ids:
                pred_binary.append(1)
                pred_prob.append(pred_ett[pred_ett["image_id"] == i]["prob"].values[0])
                pred_coord.append(pred_ett[pred_ett["image_id"] == i][["x", "y"]].values[0])
            else:
                pred_binary.append(0)
                pred_coord.append([0, 0])
                pred_prob.append(0)

            # Get ground truth
            if i in gt_ids:
                gt_binary.append(1)
                gt_coord.append(gt_ett[gt_ett["image_id"] == i][["x", "y"]].values[0])
            else:
                gt_binary.append(0)
                gt_coord.append([0, 0])
        pred_binary = np.array(pred_binary)
        gt_binary = np.array(gt_binary)
        pred_prob = np.array(pred_prob)
        gt_coord = np.array(gt_coord)
        pred_coord = np.array(pred_coord)
        return pred_binary, gt_binary, pred_prob, gt_coord, pred_coord


    def filter_normal(self, gt_carina, pred_carina, gt_ett, pred_ett):
        # Get common image ids
        ids_intersection = set(gt_carina["image_id"]).intersection(set(pred_carina["image_id"]))
        ids_intersection = ids_intersection.intersection(set(gt_ett["image_id"]))
        ids_intersection = ids_intersection.intersection(set(pred_ett["image_id"]))

        # Filter for intersecting image ids
        mask = gt_carina["image_id"].isin(ids_intersection)
        gt_carina = gt_carina.loc[mask][["x", "y"]].to_numpy()
        mask = pred_carina["image_id"].isin(ids_intersection)
        pred_carina = pred_carina.loc[mask][["x", "y"]].to_numpy()
        mask = gt_ett["image_id"].isin(ids_intersection)
        gt_ett = gt_ett.loc[mask][["x", "y"]].to_numpy()
        mask = pred_ett["image_id"].isin(ids_intersection)
        pred_ett = pred_ett.loc[mask][["x", "y"]].to_numpy()

        # Compute distance
        gt_dist = np.sqrt( np.sum((gt_carina-gt_ett)**2, axis=1) )
        pred_dist = np.sqrt( np.sum((pred_carina-pred_ett)**2, axis=1) )
        return gt_dist, pred_dist


    def eval_all(self,  ett_thres_c=[0.05, 0.1, 0.15], norma_thres=[2, 7]):
        """Evaluate all metrics.
        """
        # Compute metrics for carina
        gt_carina, pred_carina = self.filter_carina(self.gt_carina, self.pred_carina)
        AP_carina = self.carina_eval(gt_carina, pred_carina)

        # Compute metrics for ett
        pred_binary, gt_binary, pred_prob, gt_ett, pred_ett = self.filter_ett(self.gt_ett, self.pred_ett)
        # breakpoint()
        AUC_ETT = self.ETT_eval(pred_binary, gt_binary, pred_prob, gt_ett, pred_ett, thres_c=ett_thres_c)

        # Compute metrics for normal vs abnormal
        gt_dist, pred_dist = self.filter_normal(self.gt_carina, self.pred_carina, self.gt_ett, self.pred_ett)
        F1 = self.normal_vs_abnormal(gt_dist, pred_dist, thres=norma_thres)

        combined = AP_carina + AUC_ETT + F1
        # combined = AP_carina + F1
        return combined, AP_carina, AUC_ETT, F1
        # return combined, AP_carina, F1


    def carina_eval(self, gt, pred, thres=[0.5], verbose=False):
        """Compute the precision of carina detection.

        Args:
            gt: numpy array of gt coords
            preds: numpy array of pred coords
            thres: distance threshold (in cm) for a pred to be considered as correct,
                   thres can be list, and the output will be mAP instead of AP.

        Outputs:
            AP: float [or mAP if thres is a list]
        """
        precision_list = []
        for x in thres:
            TP, FP = 0, 0
            for i in range(len(gt)):
                if np.linalg.norm(gt[i] - pred[i]) < x:
                    TP += 1
                else:
                    FP += 1
            precision = TP / (TP + FP + 0.00001)
            precision_list.append(precision)
            if verbose:
                print(f"Precision for threshold {x} is {precision}")
        return np.mean(precision_list)

    def simplified_precision_recall_curve(self, y_true, y_scores, additional_filter=None):
        # Ensure y_true and y_scores are numpy arrays
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        additional_filter = np.array(additional_filter)

        # Sort prediction scores and true labels based on scores in descending order
        desc_score_indices = np.argsort(y_scores)[::-1]
        y_scores = y_scores[desc_score_indices]
        y_true = y_true[desc_score_indices]

        # If additional feature (e.g., thresholding based on error) is provided
        if additional_filter is not None:
            additional_filter = np.array(additional_filter)
            additional_filter = additional_filter[desc_score_indices]

        # Compute true positive and false positive rates
        tp = np.cumsum(y_true & additional_filter)  # True positives
        fp = np.cumsum(~y_true & additional_filter)  # False positives

        precision = tp / (tp + fp + 0.00001)
        recall = tp / np.sum(y_true)

        # Add an additional data point at the beginning
        precision = np.r_[1, precision]
        recall = np.r_[0, recall]

        return recall, precision, y_scores

    def ETT_eval(self, pred_binary, gt_binary, pred_prob, gt_coord, pred_coord, thres_c, thres_d=[0.5], verbose=False):
        """Compute the AUC of ETT detection.

        Args:
            pred_binary: list of pred binary label.
            gt_binary: list of gt binary label.
            pred_prob: list of pred probability.
            gt_coord: list of gt coords.
            pred_coord: list of pred coords.
            thres_c: confidence threshold for a pred to be considered as correct.
            thres_d: distance threshold (in cm) for a pred to be considered as correct,
                     thres can be list, and the output will be mAP instead of AP.

        Outputs:
            AUC: float
        """
        # breakpoint()
        errors = np.linalg.norm(gt_coord - pred_coord, axis=1)
        recall, precision, _ = self.simplified_precision_recall_curve(gt_binary, pred_prob, errors < thres_d[0])
        auc = metrics.auc(recall, precision)
        return auc


    def normal_vs_abnormal(self, gt, pred, thres=[2,7]):
        """Compute the F1 score of normal vs abnormal detection.

        Args:
            gt: list of gt distance
            preds: list of pred distance
            thres: list of 2 values for distance threshold (in cm) for a pred to be
                   considered as normal, the 2 values are lower and upper bounds.

        Outputs:
            F1: float [or mAP if thres is a list]
        """
        TP, FP, FN, TN = 0, 0, 0, 0
        for i in range(len(gt)):
            if gt[i] > thres[0] and gt[i] < thres[1]:
                if pred[i] > thres[0] and pred[i] < thres[1]:
                    TP += 1
                else:
                    FN += 1
            else:
                if pred[i] > thres[0] and pred[i] < thres[1]:
                    FP += 1
                else:
                    TN += 1
        precision = TP / (TP + FP + 0.00001)
        recall = TP / (TP + FN + 0.00001)
        F1 = 2 * precision * recall / (precision + recall + 0.00001)
        return F1


if __name__ == "__main__":
    pass
    # metric = UpdatedMetric(
    #     gt_file = "GT_test/martin_test.csv",
    #     pred_file = "GT_test/martin_test.csv"
    # )
    # values = metric.eval_all()
    # print(values)

    # # dummy data for the evaluation metric
    # # carina
    # gt_carina = np.random.rand(100, 2)
    # pred_carina = np.random.rand(100, 2)
    # # ETT
    # gt_binary = np.random.randint(2, size=100)
    # pred_prob = np.random.rand(100)
    # gt_ett = np.random.rand(100, 2)
    # pred_ett = np.random.rand(100, 2)
    # # distance
    # gt_dist = np.linalg.norm(gt_ett - gt_carina, axis=1)
    # pred_dist = np.linalg.norm(pred_ett - pred_carina, axis=1)

    # # unit test for carina_eval
    # print(metric.carina_eval(gt_carina, pred_carina))

    # # unit test for ETT_eval
    # print(metric.ETT_eval(gt_binary, pred_prob, gt_ett, pred_ett, \
    #                       thres_c=[0.3, 0.5, 0.7]))

    # # unit test for normal vs abnormal
    # print(metric.normal_vs_abnormal(gt_dist, pred_dist, thres=[0.2, 0.7]))