import pandas as pd
import re


poma_3class = pd.read_csv('../../dataset/real_poma_3class_dataset_210518.csv')
updrs_3class = pd.read_csv('../../dataset/real_updrs_3class_dataset_210518.csv')

poma_dataset_3class = poma_3class.copy()
updrs_dataset_3class = updrs_3class.copy()

poma_cols = [re.sub(r'[\W_]', "", i) for i in poma_dataset_3class.columns]

for i in range(len(poma_cols)):
    poma_cols[i] = poma_cols[i] + str(i)

poma_dataset_3class.columns = poma_cols

updrs_cols = [re.sub(r'[\W_]', "", i) for i in updrs_dataset_3class.columns]

for i in range(len(updrs_cols)):
    updrs_cols[i] = updrs_cols[i] + str(i)

updrs_dataset_3class.columns = updrs_cols


print(poma_dataset_3class)
print(updrs_dataset_3class)

poma_dataset_3class.to_csv('non_str_poma_3class_dataset_210518.csv', index=False)
updrs_dataset_3class.to_csv('non_str_updrs_3class_dataset_210518.csv', index=False)


