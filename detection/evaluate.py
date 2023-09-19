from read_outputs import get_pred_labels

import sys
sys.path.append('../../ETT_Evaluation')
from eval.eval import ETTEvaler
from eval.updated_eval import UpdatedMetric
from visualize.visualize import Visualization
import numpy as np
import pandas as pd
from tabulate import tabulate


class ReadEval():
    def __init__(self, gt_file, pred_file, pixel_spacing_file,
                 resized_dim, encode, verbose=False):
        self.gt_file = gt_file
        self.pred_file = pred_file
        self.pixel_spacing_file = pixel_spacing_file
        self.resized_dim = resized_dim
        self.encode = encode
        self.verbose = verbose



    def read_combined_metric(self):
        updated_evaler = UpdatedMetric(self.gt_file, self.pred_file,
                                       self.resized_dim,
                                       self.pixel_spacing_file,
                                       self.encode)
        values = updated_evaler.eval_all()

        print(f"combined: {values[0]}")
        print(f"AP_carina: {values[1]}")
        print(f"AUC_ETT: {values[2]}")
        print(f"F1_normal_abnormal: {values[3]}")
        return values

    def read_eval(self):
        evaler = ETTEvaler(self.gt_file, self.pred_file,
                           self.resized_dim, encode=self.encode,
                           pixel_spacing_file=self.pixel_spacing_file,
                           verbose=self.verbose)

        tp, fp, tn, fn, distances, id_to_dis = evaler.gt_pred_distance(target='tip')
        evaler.get_metrics(tp, fp, tn, fn, distances, cat='tip')
        sorted_dis = list(sorted(id_to_dis.items(), key=lambda x:x[1]))
        min_ids = [i[0] for i in sorted_dis[:10]]
        max_ids = [i[0] for i in sorted_dis[-10:]]
        return min_ids, max_ids, np.mean(distances)


if __name__=="__main__":
    mimic = False
    # hospitals = ["Cedars-Sinai", "Chiang_Mai_University", "Morales_Meseguer_Hospital",
    #              "Newark_Beth_Israel_Medical_Center", "NYU_Langone_Health",
    #              "Osaka_City_University", "Technical_University_of_Munich",
    #              "Universitätsklinikum_Tübingen", "University_of_Miami"]
    hospitals = ["Newark_Beth_Israel_Medical_Center"]
    resized_dim = 1280
    overall = []
    for hospital_name in hospitals:
        print(hospital_name)
        gt_file = f"labels/gt_labels_{hospital_name}.csv"
        pred_file = f"labels/pred_labels_{hospital_name}.csv"
        pred_json = f"metric_v2_preds/metric_v2_{hospital_name}_bright.bbox.json"


        pixel_spacing_file = "/home/cat302/ETT-Project/ETT_Evaluation/pixel_spacing_10_hospitals_cleaned.csv"
        encode={"carina": 3046, "tip": 3047}
        if mimic:
            pixel_spacing_file = "/home/cat302/ETT-Project/ETT_Evaluation/pixel_spacing.csv"
            encode={"carina": 0, "tip": 1}

        get_pred_labels(pred_json, pred_file)
        read_eval = ReadEval(gt_file, pred_file, pixel_spacing_file,
                            resized_dim, encode, verbose=False)

        values = read_eval.read_combined_metric()
        min_ids, max_ids, mean_dis = read_eval.read_eval()
        row = [hospital_name]
        row.extend(values)
        row.append(mean_dis)
        overall.append(row)

        ### Visualzie good and bad examples
        # good_dir = f'/n/data1/hms/dbmi/rajpurkar/lab/ett/{hospital_name}_good_examples'
        # bad_dir = f'/n/data1/hms/dbmi/rajpurkar/lab/ett/{hospital_name}_bad_examples'
        # good_ids, bad_ids = read_eval.read_eval()
        # visualization = Visualization(
        #     gt_anno=f'/n/data1/hms/dbmi/rajpurkar/lab/ett/hospital_downsized/{hospital_name}/annotations/annotations.json',
        #     pred_anno=pred_json,
        #     image_dir=f'/n/data1/hms/dbmi/rajpurkar/lab/ett/hospital_downsized/{hospital_name}/images_bright'
        # )
        # visualization.show_images(
        #     good_ids, category_id=3047,
        #     save_dir=good_dir
        # )

        # visualization.show_images(
        #     bad_ids, category_id=3047,
        #     save_dir=bad_dir
        # )

    overall = pd.DataFrame(overall, columns=['hospital', 'combined',
    'AP_carina', 'AUC_ETT', 'F1_normal_abnormal', 'mean_tip_distance'])
    # display(overall)
    print(tabulate(overall, headers='keys', tablefmt='psql'))