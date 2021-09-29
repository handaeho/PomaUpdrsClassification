import numpy as np
import pandas as pd


# dataset_3class = pd.read_csv('dataset/poma_updrs_3class_dataset_210429.csv')
# print(dataset_3class)
#
# dataset_3class['poma_danger_3class'] = np.nan
# dataset_3class['updrs_danger_3class'] = np.nan
#
# for i in range(len(dataset_3class)):
#     if 0 <= dataset_3class['POMA Total'][i] <= 18:
#         dataset_3class['poma_danger_3class'][i] = 2
#     elif 19 <= dataset_3class['POMA Total'][i] <= 24:
#         dataset_3class['poma_danger_3class'][i] = 1
#     elif 25 <= dataset_3class['POMA Total'][i] <= 28:
#         dataset_3class['poma_danger_3class'][i] = 0
#
# for i in range(len(dataset_3class)):
#     if 0 <= dataset_3class['UPDRS Total'][i] <= 5:
#         dataset_3class['updrs_danger_3class'][i] = 0
#     elif 6 <= dataset_3class['UPDRS Total'][i] <= 14:
#         dataset_3class['updrs_danger_3class'][i] = 1
#     elif 15 <= dataset_3class['UPDRS Total'][i]:
#         dataset_3class['updrs_danger_3class'][i] = 2
#
#
# print(dataset_3class)
#
# dataset_3class.to_csv('real_poma_updrs_3class_dataset_210518.csv', index=False)

dataset_2class = pd.read_csv('../../dataset/poma_updrs_3class_dataset_210429.csv')
print(dataset_2class)

dataset_2class['poma_danger_2class'] = np.nan
dataset_2class['updrs_danger_2class'] = np.nan

for i in range(len(dataset_2class)):
    if 0 <= dataset_2class['POMA Total'][i] <= 24:
        dataset_2class['poma_danger_2class'][i] = 1
    elif 25 <= dataset_2class['POMA Total'][i] <= 28:
        dataset_2class['poma_danger_2class'][i] = 0

for i in range(len(dataset_2class)):
    if 0 <= dataset_2class['UPDRS Total'][i] <= 5:
        dataset_2class['updrs_danger_2class'][i] = 0
    elif 6 <= dataset_2class['UPDRS Total'][i]:
        dataset_2class['updrs_danger_2class'][i] = 1


print(dataset_2class)

dataset_2class.to_csv('real_poma_updrs_2class_dataset_210622.csv', index=False)
