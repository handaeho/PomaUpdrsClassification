import pandas as pd
import numpy as np

np.random.seed(1234)

poma_3class = pd.read_csv('../../dataset/real_poma_2class_dataset_210518.csv')
updrs_3class = pd.read_csv('../../dataset/real_updrs_2class_dataset_210518.csv')

poma_dataset = poma_3class.copy()
updrs_dataset = updrs_3class.copy()

poma_1500 = poma_dataset.sample(n=1500)
updrs_1500 = updrs_dataset.sample(n=1500)

poma_1500 = poma_1500.reset_index(drop=True)
updrs_1500 = updrs_1500.reset_index(drop=True)

print(poma_1500)
print(updrs_1500)

poma_1500.to_csv('poma_dataset_210518_1500.csv', index=False)
updrs_1500.to_csv('updrs_dataset_210518_1500.csv', index=False)
