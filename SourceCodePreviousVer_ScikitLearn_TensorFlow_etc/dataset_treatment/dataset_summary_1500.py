import pandas as pd
from sklearn.model_selection import train_test_split


poma_df = pd.read_csv('dataset/poma_dataset_210504_1500.csv')
print(poma_df)  # [19786 rows x 96 columns]

updrs_df = pd.read_csv('dataset/updrs_dataset_210504_1500.csv')
print(updrs_df)  # [19786 rows x 96 columns]

poma_dataset = poma_df.copy()
updrs_dataset = updrs_df.copy()

poma_features = poma_dataset
poma_labels = poma_dataset.pop('poma_danger_3class')

updrs_features = updrs_dataset
updrs_labels = updrs_dataset.pop('updrs_danger_3class')

poma_x_train, poma_x_test, poma_y_train, poma_y_test = train_test_split(poma_features, poma_labels,
                                                                        test_size=0.2,
                                                                        stratify=poma_labels,
                                                                        shuffle=True,
                                                                        random_state=1234)

updrs_x_train, updrs_x_test, updrs_y_train, updrs_y_test = train_test_split(updrs_features, updrs_labels,
                                                                            test_size=0.2,
                                                                            stratify=updrs_labels,
                                                                            shuffle=True,
                                                                            random_state=1234)

print('학습 데이터', poma_x_train.shape, poma_y_train.shape)            # 학습 데이터 (1200, 95) (1200,)
print('테스트 데이터', poma_x_test.shape, poma_y_test.shape)             # 테스트 데이터 (300, 95) (300,)

print('학습 데이터', updrs_x_train.shape, updrs_y_train.shape)           # 학습 데이터 (1200, 95) (1200,)
print('테스트 데이터', updrs_x_test.shape, updrs_y_test.shape)            # 테스트 데이터 (300, 95) (300,)

print(len(poma_df[poma_df['poma_danger_3class'] == 0]))  # 218
print(len(poma_df[poma_df['poma_danger_3class'] == 1]))  # 545
print(len(poma_df[poma_df['poma_danger_3class'] == 2]))  # 737

print(len(updrs_df[updrs_df['updrs_danger_3class'] == 0]))  # 690
print(len(updrs_df[updrs_df['updrs_danger_3class'] == 1]))  # 571
print(len(updrs_df[updrs_df['updrs_danger_3class'] == 2]))  # 239

# poma
print(poma_y_train.value_counts())      # 2    590 / 1    436 / 0    174
print(poma_y_test.value_counts())       # 2    147 / 1    109 / 0    44

# updrs
print(updrs_y_train.value_counts())     # 0    552 / 1    457 / 2    191
print(updrs_y_test.value_counts())      # 0    138 / 1    114 / 2    48




