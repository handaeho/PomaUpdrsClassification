import pandas as pd
from sklearn.model_selection import train_test_split

# poma_df = pd.read_csv('dataset/real_poma_3class_dataset_210518.csv')
poma_df = pd.read_csv('../../dataset/real_poma_2class_dataset_210518.csv')
print(poma_df)  # [19786 rows x 96 columns]

# updrs_df = pd.read_csv('dataset/real_updrs_3class_dataset_210518.csv')
updrs_df = pd.read_csv('../../dataset/real_updrs_2class_dataset_210518.csv')
print(updrs_df)  # [19786 rows x 96 columns]

poma_dataset = poma_df.copy()
updrs_dataset = updrs_df.copy()

poma_features = poma_dataset
# poma_labels = poma_dataset.pop('poma_danger_3class')
poma_labels = poma_dataset.pop('poma_danger_2class')

updrs_features = updrs_dataset
# updrs_labels = updrs_dataset.pop('updrs_danger_3class')
updrs_labels = updrs_dataset.pop('updrs_danger_2class')

poma_x_train, poma_x_test, poma_y_train, poma_y_test = train_test_split(poma_features, poma_labels,
                                                                        test_size=0.2,
                                                                        stratify=poma_labels,
                                                                        shuffle=True,
                                                                        random_state=1234)

poma_x_train_t, poma_x_eval, poma_y_train_t, poma_y_eval = train_test_split(poma_x_train, poma_y_train,
                                                                            test_size=0.2,
                                                                            stratify=poma_y_train,
                                                                            shuffle=True,
                                                                            random_state=1234)

updrs_x_train, updrs_x_test, updrs_y_train, updrs_y_test = train_test_split(updrs_features, updrs_labels,
                                                                            test_size=0.2,
                                                                            stratify=updrs_labels,
                                                                            shuffle=True,
                                                                            random_state=1234)

updrs_x_train_t, updrs_x_eval, updrs_y_train_t, updrs_y_eval = train_test_split(updrs_x_train, updrs_y_train,
                                                                                test_size=0.2,
                                                                                stratify=updrs_y_train,
                                                                                shuffle=True,
                                                                                random_state=1234)

print('학습 데이터', poma_x_train.shape, poma_y_train.shape)            # 학습 데이터 (15828, 95) (15828,)
print('실제 학습 데이터', poma_x_train_t.shape, poma_y_train_t.shape)    # 실제 학습 데이터 (12662, 95) (12662,)
print('평가 데이터', poma_x_eval.shape, poma_y_eval.shape)              # 평가 데이터 (3166, 95) (3166,)
print('테스트 데이터', poma_x_test.shape, poma_y_test.shape)             # 테스트 데이터 (3958, 95) (3958,)

print('학습 데이터', updrs_x_train.shape, updrs_y_train.shape)           # 학습 데이터 (15828, 95) (15828,)
print('실제 학습 데이터', updrs_x_train_t.shape, updrs_y_train_t.shape)   # 실제 학습 데이터 (12662, 95) (12662,)
print('평가 데이터', updrs_x_eval.shape, updrs_y_eval.shape)             # 평가 데이터 (3166, 95) (3166,)
print('테스트 데이터', updrs_x_test.shape, updrs_y_test.shape)           # 테스트 데이터 (3958, 95) (3958,)

# print(len(poma_df[poma_df['poma_danger_3class'] == 0]))  # 9936
# print(len(poma_df[poma_df['poma_danger_3class'] == 1]))  # 6700
# print(len(poma_df[poma_df['poma_danger_3class'] == 2]))  # 3150

# print(len(updrs_df[updrs_df['updrs_danger_3class'] == 0]))  # 9058
# print(len(updrs_df[updrs_df['updrs_danger_3class'] == 1]))  # 7639
# print(len(updrs_df[updrs_df['updrs_danger_3class'] == 2]))  # 3089

print(len(poma_df[poma_df['poma_danger_2class'] == 0]))  # 9936
print(len(poma_df[poma_df['poma_danger_2class'] == 1]))  # 6700

print(len(updrs_df[updrs_df['updrs_danger_2class'] == 0]))  # 9058
print(len(updrs_df[updrs_df['updrs_danger_2class'] == 1]))  # 7639

# poma
print(poma_y_train.value_counts())      # 0    7948 / 1    5360 / 2    2520
print(poma_y_train_t.value_counts())    # 0    6358 / 1    4288 / 2    2016
print(poma_y_eval.value_counts())       # 0    1590 / 1    1072 / 2     504
print(poma_y_test.value_counts())       # 0    1988 / 1    1340 / 2     630

# updrs
print(updrs_y_train.value_counts())     # 0    7246 / 1    6111 / 2    2471
print(updrs_y_train_t.value_counts())   # 0    5796 / 1    4889 / 2    1977
print(updrs_y_eval.value_counts())      # 0    1450 / 1    1222 / 2     494
print(updrs_y_test.value_counts())      # 0    1812 / 1    1528 / 2     618

