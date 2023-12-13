from PIL import Image
import os
import numpy as np
import json
import pandas as pd
import math
from matplotlib import pyplot as plt
from ast import literal_eval

hospitals = [
    # 'Ascension-Seton', 'Cedars-Sinai','Chiang_Mai_University', 
    #         'Fundación_Santa_Fe_de_Bogotá', 'Lawson_Health', 
            # 'Morales_Meseguer_Hospital', 
            # 'National_University_of_Singapore',
            'Newark_Beth_Israel_Medical_Center', 'NYU_Langone_Health',
            'Osaka_City_University', 'Rhode_Island_Hospital', 
            'Sunnybrook_Research_Institute', 'Technical_University_of_Munich',
            'Universitätsklinikum_Essen', 'Universitätsklinikum_Tübingen', 
            'University_of_Miami']
def plot_image(img, row, outpath):
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(img, cmap = 'gray')
    # breakpoint()
    # ax.scatter(*zip(*points[:,::-1]), alpha = 0.1, c = 'r', s = 3, label = 'Predicted Endotracheal tube')
    ax.scatter(row['GT ETT'].values[0][0], row['GT ETT'].values[0][1], c = 'purple', s = 5, alpha = 1, label = 'GT Endotracheal tube tip')
    ax.scatter(row['GT Carina'].values[0][0], row['GT Carina'].values[0][1], c = 'green', s = 5, alpha = 1, label = 'GT Carina')
    ax.scatter(row['Pred ETT'].values[0][0], row['Pred ETT'].values[0][1], c = 'pink', s = 5, alpha = 1, label = 'Pred Endotracheal tube tip')
    ax.scatter(row['Pred Carina'].values[0][0], row['Pred Carina'].values[0][1], c = 'blue', s = 5, alpha = 1, label = 'Pred Carina')
    gt_ett_carina_dis = row['GT Tip-Carina Distance'].values[0] if not pd.isna(row['GT Tip-Carina Distance'].values) else np.nan
    # gt_ett_carina_dis = row['GT Tip-Carina Distance'].iloc[0] if not pd.isna(row['GT Tip-Carina Distance']).any() else np.nan
    # gt_ett_carina_dis = row['GT Tip-Carina Distance'][0] if not pd.isna(row['GT Tip-Carina Distance'].iloc[0]) else np.nan

    pred_ett_carina_dis = row['Pred Tip-Carina Distance'].values[0] if not pd.isna(row['Pred Tip-Carina Distance'].values) else np.nan
    ax.annotate(f'GT ETT-carina distance = {gt_ett_carina_dis:.2f}cm.\nPredicted ETT-carina distance = {pred_ett_carina_dis:.2f}cm',
                xy=(0.2, 0.05), xycoords='axes fraction', fontsize=8,
                bbox=dict(facecolor='none', edgecolor='k', pad=3), color = 'k')
    ax.legend(fontsize=6)
    fig.savefig(os.path.join(outpath, f"{row['Image'].values[0]}.png"), bbox_inches="tight", pad_inches=0,
                dpi = 500)
    plt.close(fig)
    
def plot(image_path, report, outpath):
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    for row in report.loc[report['GT ETT'].notnull(), 'GT ETT'].index:
        report.at[row, 'GT ETT'] = literal_eval(report.at[row, 'GT ETT'])
    for row in report.loc[report['GT ETT'].isnull(), 'GT ETT'].index:
        report.at[row, 'GT ETT'] = [np.nan, np.nan]

    for row in report.loc[report['GT Carina'].notnull(), 'GT Carina'].index:
        report.at[row, 'GT Carina'] = literal_eval(report.at[row, 'GT Carina'])
    for row in report.loc[report['GT Carina'].isnull(), 'GT Carina'].index:
        report.at[row, 'GT Carina'] = [np.nan, np.nan]

    for row in report.loc[report['Pred ETT'].notnull(), 'Pred ETT'].index:
        report.at[row, 'Pred ETT'] = literal_eval(report.at[row, 'Pred ETT'])
    for row in report.loc[report['Pred ETT'].isnull(), 'Pred ETT'].index:
        report.at[row, 'Pred ETT'] = [np.nan, np.nan]

    for row in report.loc[report['Pred Carina'].notnull(), 'Pred Carina'].index:
        report.at[row, 'Pred Carina'] = literal_eval(report.at[row, 'Pred Carina'])
    for row in report.loc[report['Pred Carina'].isnull(), 'Pred Carina'].index:
        report.at[row, 'Pred Carina'] = [np.nan, np.nan]
        
    for image in os.listdir(image_path):
        row = report[report['Image']==image[:-4]]
        if row.empty:
            print(f'No report found for image {image}')
            continue
        img = Image.open(os.path.join(image_path, image))
        plot_image(img, row, outpath)

if __name__ == '__main__':
    for hospital in hospitals:
        print(hospital)
        image_path = f'/n/data1/hms/dbmi/rajpurkar/lab/MAIDA_ETT/hospital_downsized_new/{hospital}/images'
        report = pd.read_csv(f'/n/scratch3/users/c/cat302/ETT-Project/InternImage/detection/labels/sheets/{hospital}/err_report.csv')
        outpath = f'/n/scratch3/users/c/cat302/ETT-Project/InternImage/detection/labels/sheets/{hospital}/visual'
        plot(image_path, report, outpath)