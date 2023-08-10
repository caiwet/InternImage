from read_outputs import get_pred_labels

import sys
sys.path.append('../../ETT_Evaluation')
from eval.eval import ETTEvaler
from eval.updated_eval import UpdatedMetric
import numpy as np

class ReadEval():
    def __init__(self, gt_file, pred_file, pixel_spacing_file,
                 resized_dim, encode, verbose=False):
        self.gt_file = gt_file
        self.pred_file = pred_file
        self.pixel_spacing_file = pixel_spacing_file
        self.resized_dim = resized_dim
        self.encode = encode
        self.verbose = verbose

        self.read_combined_metric()
        self.read_eval()

    def read_combined_metric(self):
        updated_evaler = UpdatedMetric(self.gt_file, self.pred_file,
                                       self.pixel_spacing_file,
                                       self.resized_dim, self.encode)
        values = updated_evaler.eval_all()
        print(values)
        print(f"combined: {values[0]}")
        print(f"AP_carina: {values[1]}")
        print(f"AUC_ETT: {values[2]}")
        print(f"F1_normal_abnormal: {values[3]}")

    def read_eval(self):
        evaler = ETTEvaler(self.gt_file, self.pred_file,
                           self.resized_dim, encode=self.encode,
                           pixel_spacing_file=self.pixel_spacing_file,
                           verbose=self.verbose)

        tp, fp, tn, fn, distances = evaler.gt_pred_distance(target='tip')
        evaler.get_metrics(tp, fp, tn, fn, distances, cat='tip')


if __name__=="__main__":
    mimic = False
    hospital_name = "NYU_Langone_Health"
    gt_file = f"labels/gt_labels_{hospital_name}.csv"
    pred_file = f"labels/pred_labels_{hospital_name}.csv"
    pred_json = f'metric_v2_{hospital_name}.bbox.json'
    resized_dim = 1280


    pixel_spacing_file = '/home/cat302/ETT-Project/ETT_Evaluation/pixel_spacing_10_hospitals.csv'
    encode={"carina": 3046, "tip": 3047}
    if mimic:
        pixel_spacing_file = '/home/cat302/ETT-Project/ETT_Evaluation/pixel_spacing.csv'
        encode={"carina": 0, "tip": 1}

    get_pred_labels(pred_json, pred_file)
    read_eval = ReadEval(gt_file, pred_file, pixel_spacing_file,
                         resized_dim, encode, verbose=False)

