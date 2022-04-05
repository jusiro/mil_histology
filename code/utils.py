import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def cifar10_test_dataset(dir_dataset):

    files = os.listdir(dir_dataset)

    Y = []
    X = []
    for iFile in files:
        if 'Other' in iFile:
            y = 0
        else:
            y = iFile.split('_')[-2][-1]

    return X, Y


def classification_report_csv(report, dir_out):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_xlsx(dir_out + 'classification_report.xlsx', index = False)


def plot_image(x, norm_intensity=False):
    # channel first
    x = np.transpose(x, (1, 2, 0))
    if norm_intensity:
        x = x / 255.0

    plt.imshow(x)
    plt.axis('off')
    plt.show()
