import numpy as np
import json
import pandas as pd
import math
import matplotlib.pyplot as plt
from PIL import Image
import os
import scipy.stats as stats

def id_to_name(gt_annotation):
    with open(gt_annotation) as f:
        gt_data = json.load(f)

    id_to_filename = {}
    for img in gt_data["images"]:
        id_to_filename[img['id']] = img['file_name'][:-4]
    return id_to_filename

def pixel_to_cm(img_id, dis, pixel_spacing, resized_dim=1280):
    tmp = pixel_spacing[pixel_spacing['image_id']==img_id]
    if len(tmp) == 0:
        return dis*2500*0.139*0.1/resized_dim
    cropped_size = min(int(tmp['original_width']), int(tmp['original_height']))
    ps = cropped_size * float(tmp['pixel_spacing_x']) / resized_dim  ## Assume x and y have the same pixel spacing

    return ps * dis * 0.1

def get_spread_sheet(gt_labels, pred_labels, gt_annotation):
    gt = pd.read_csv(gt_labels)
    pred = pd.read_csv(pred_labels)

    df1 = gt[gt['category_id'] == 3047][['image_id', 'x', 'y']]
    df1['GT ETT'] = df1.apply(lambda row: [row['x'], row['y']], axis=1)
    df1 = df1.drop(['x', 'y'], axis=1)

    df2 = gt[gt['category_id'] == 3046][['image_id', 'x', 'y']]
    df2['GT Carina'] = df2.apply(lambda row: [row['x'], row['y']], axis=1)
    df2 = df2.drop(['x', 'y'], axis=1)

    result_df = pd.merge(df1, df2, on='image_id', how='outer')

    df1 = pred[pred['category_id'] == 3047][['image_id', 'x', 'y']]
    df1['Pred ETT'] = df1.apply(lambda row: [row['x'], row['y']], axis=1)
    df1 = df1.drop(['x', 'y'], axis=1)

    df2 = pred[pred['category_id'] == 3046][['image_id', 'x', 'y']]
    df2['Pred Carina'] = df2.apply(lambda row: [row['x'], row['y']], axis=1)
    df2 = df2.drop(['x', 'y'], axis=1)

    result_df = pd.merge(result_df, df1, on='image_id', how='outer')
    result_df = pd.merge(result_df, df2, on='image_id', how='outer')
    id_to_filename = id_to_name(gt_annotation)
    result_df['Image'] = result_df['image_id'].map(id_to_filename)
    result_df = get_dis_report(result_df)
    return result_df

def plot_dis(data, hospital, title, outfile):
    # Create a histogram
    fig, ax = plt.subplots()
    ax.hist(data, bins=20, edgecolor='k', alpha=0.7)
    ax.set_xlabel('Error')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{hospital} {title}')

    # Calculate quantiles
    q25 = np.nanpercentile(data, 25)
    q75 = np.nanpercentile(data, 75)

    # Display quantiles
    ax.axvline(q25, color='r', linestyle='dashed', linewidth=2, label='25th Quantile')
    ax.axvline(q75, color='g', linestyle='dashed', linewidth=2, label='75th Quantile')
    ax.legend()

    fig.savefig(f'/n/scratch3/users/c/cat302/ETT-Project/InternImage/detection/labels/sheets/{hospital}/{outfile}.png', dpi=300, bbox_inches='tight')

def plot_cumulative_err(data, hospital):
    # Sort the DataFrame by the 'tip-carina error' column
    df = data.copy()
    df = df.sort_values(by='Tip-Carina Error')

    # Create a cumulative count of items less than or equal to each error
    cumulative_count = [i + 1 for i in range(len(df))]

    fig, ax = plt.subplots()

    # Plot the cumulative count against the error values
    ax.plot(df['Tip-Carina Error'], cumulative_count)

    # Set labels and title
    ax.set_xlabel('Error Value')
    ax.set_ylabel('Cumulative Count')
    ax.set_title(f'{hospital} Abs. T-C Error between GT and Pred')

    # Show the plot
    fig.savefig(f'/n/scratch3/users/c/cat302/ETT-Project/InternImage/detection/labels/sheets/{hospital}/cum_err.png')

def euclidean_distance(point1, point2):
    if point1 is np.nan or point2 is np.nan:
        return np.nan
    x1, y1 = point1
    x2, y2 = point2
    if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
        return np.nan
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def get_dis_report(df, pixel_spacing_file = "/home/cat302/ETT-Project/data_tools/pixel_spacing_17_hospitals.csv",
                   resized_dim = 1280):
    pixel_spacing = pd.read_csv(pixel_spacing_file)
    pixel_spacing['scale'] = pixel_spacing.apply(
        lambda row: 0.1 * min(int(row['original_width']), int(row['original_height'])) * float(row['pixel_spacing_x']) / resized_dim, ## Assume x and y have the same pixel spacing
        axis=1  
    )
    pixel_spacing = pixel_spacing[['image','scale']]


    df = pd.merge(df, pixel_spacing, left_on='Image', right_on='image', how='inner')

    df['Tip Prediction Error'] = df.apply(
        lambda row: euclidean_distance(row['GT ETT'], row['Pred ETT']), 
        axis=1
    )
    df['Tip Prediction Error'] = df['Tip Prediction Error'] * df['scale'] 

    df['Carina Prediction Error'] = df.apply(
        lambda row: euclidean_distance(row['GT Carina'], row['Pred Carina']),
        axis=1
    )
    df['Carina Prediction Error'] = df['Carina Prediction Error'] * df['scale'] 

    df['GT Tip-Carina Distance'] = df.apply(
        lambda row: euclidean_distance(row['GT Carina'], row['GT ETT']),
        axis=1
    )
    df['GT Tip-Carina Distance'] = df['GT Tip-Carina Distance'] * df['scale']

    df['Pred Tip-Carina Distance'] = df.apply(
        lambda row: euclidean_distance(row['Pred Carina'], row['Pred ETT']),
        axis=1
    )
    df['Pred Tip-Carina Distance'] = df['Pred Tip-Carina Distance'] * df['scale']
    df['Tip-Carina Error'] = (df['GT Tip-Carina Distance'] - df['Pred Tip-Carina Distance']).abs()
    return df

if __name__ == "__main__":
    hospitals = ['Ascension-Seton', 'Cedars-Sinai','Chiang_Mai_University', 
                 'Fundación_Santa_Fe_de_Bogotá', 'Lawson_Health', 
                 'Morales_Meseguer_Hospital', 'National_University_of_Singapore',
                 'Newark_Beth_Israel_Medical_Center', 'NYU_Langone_Health',
                 'Osaka_City_University', 'Rhode_Island_Hospital', 
                 'Sunnybrook_Research_Institute', 'Technical_University_of_Munich',
                 'Universitätsklinikum_Essen', 'Universitätsklinikum_Tübingen', 
                 'University_of_Miami']
    
    # for hospital in hospitals:
    #     print(hospital)
    #     if not os.path.exists(f'gt_labels_{hospital}.csv'):
    #         print('skip')
    #         continue
    #     outpath = f'/n/scratch3/users/c/cat302/ETT-Project/InternImage/detection/labels/sheets/{hospital}'
    #     if not os.path.exists(outpath):
    #         os.makedirs(outpath)

    #     result_df = get_spread_sheet(f'gt_labels_{hospital}.csv', f'pred_labels_{hospital}.csv',
    #                      f'/n/data1/hms/dbmi/rajpurkar/lab/MAIDA_ETT/hospital/{hospital}/annotations/annotations.json')
    #     plot_cumulative_err(result_df, hospital)

    #     plot_dis(result_df['Tip Prediction Error'], hospital, title='Histogram of Euclidean Tip Error',
    #              outfile='histogram')
    #     tmp = result_df.dropna(axis=0)
    #     y_dis = [row['scale'] * (row['Pred ETT'][1] - row['GT ETT'][1]) for index, row in tmp.iterrows()]
    #     plot_dis(y_dis, hospital, title='Histogram of Tip Position Error (Pred - GT)',
    #              outfile='y_err')
        
    #     result_df = result_df[['image_id', 'Image', 'GT ETT', 'Pred ETT', 
    #     'GT Carina', 'Pred Carina', 'Tip Prediction Error',
    #     'Carina Prediction Error', 'GT Tip-Carina Distance',
    #     'Pred Tip-Carina Distance', 'Tip-Carina Error']]
        
    #     result_df.to_csv(f'{outpath}/err_report.csv', index=False)

    y_dis_all = []
    fig, ax = plt.subplots()
    for hospital in hospitals:
        result_df = get_spread_sheet(f'gt_labels_{hospital}.csv', f'pred_labels_{hospital}.csv',
                         f'/n/data1/hms/dbmi/rajpurkar/lab/MAIDA_ETT/hospital/{hospital}/annotations/annotations.json')
        result_df = result_df.dropna(axis=0)
        y_dis = [row['scale'] * (row['Pred ETT'][1] - row['GT ETT'][1]) for index, row in result_df.iterrows()]
        y_dis_all.extend(y_dis)
        ax.hist(result_df['Tip-Carina Error'], bins=20, edgecolor='k', alpha=0.3, label=hospital)
    ax.legend()
    ax.set_xlabel("Tip-Carina Error")
    ax.set_ylabel("Frequency")
    ax.set_title("InternImage")
    fig.savefig("/n/scratch3/users/c/cat302/ETT-Project/InternImage/detection/labels/sheets/results/overall_t-c_err.png")
    
    # print(y_dis_all)
    # # Perform the Shapiro-Wilk test
    # statistic, p_value = stats.shapiro(y_dis_all)

    # # Check if the data follows a normal distribution
    # alpha = 0.05  # Significance level
    # print(p_value)
    # print(statistic)
    # if p_value > alpha:
    #     print("Data appears to be normally distributed")
    # else:
    #     print("Data does not appear to be normally distributed")

