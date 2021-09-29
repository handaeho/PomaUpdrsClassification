import re

import pandas as pd
from sklearn.preprocessing import RobustScaler


def poma_load_dataset():
    poma_3class = pd.read_csv('../../dataset/real_poma_3class_dataset_210518.csv')

    poma_dataset_3class = poma_3class.copy()

    poma_dataset_3class = poma_dataset_3class.sample(n=100)

    # TF2 Pipe-line에는 컬럼명에 특수문자 불가.
    cols = [re.sub(r'[\W_]', "", i) for i in poma_dataset_3class.columns]

    for i in range(len(cols)):
        cols[i] = cols[i] + str(i)

    poma_dataset_3class.columns = cols

    # print(poma_dataset_3class)

    # 'danger' 컬럼 -> 라벨
    features = poma_dataset_3class.iloc[:, 1:].values
    labels = poma_dataset_3class['pomadanger3class0'].values

    # 변형 객체 생성 / 모수 분포 저장 (fit)
    rs_scaler = RobustScaler().fit(features)

    # 데이터 스케일링 (transform)
    features_scaled = rs_scaler.transform(features)

    features_scaled_df = pd.DataFrame(features_scaled)
    labels_df = pd.DataFrame(labels)

    concatenated_features_scaled_labels_df = pd.concat([labels_df, features_scaled_df], axis=1)

    return concatenated_features_scaled_labels_df


def updrs_load_dataset():
    updrs_3class = pd.read_csv('../../dataset/real_updrs_3class_dataset_210518.csv')

    updrs_dataset_3class_all = updrs_3class.copy()

    updrs_dataset_3class = updrs_dataset_3class_all.sample(n=100)

    # TF2 Pipe-line에는 컬럼명에 특수문자 불가.
    cols = [re.sub(r'[\W_]', "", i) for i in updrs_dataset_3class.columns]

    for i in range(len(cols)):
        cols[i] = cols[i] + str(i)

    updrs_dataset_3class.columns = cols

    # print(updrs_dataset_3class)

    # 'danger' 컬럼 -> 라벨
    features = updrs_dataset_3class.iloc[:, 1:].values
    labels = updrs_dataset_3class['updrsdanger3class0'].values

    # 변형 객체 생성
    rs_scaler = RobustScaler().fit(features)

    # 데이터 스케일링
    features_scaled = rs_scaler.transform(features)

    features_scaled_df = pd.DataFrame(features_scaled)
    labels_df = pd.DataFrame(labels)

    concatenated_features_scaled_labels_df = pd.concat([labels_df, features_scaled_df], axis=1)

    return concatenated_features_scaled_labels_df

if __name__ == '__main__':
    poma_concatenated_features_scaled_labels_df = poma_load_dataset()

    updrs_concatenated_features_scaled_labels_df = updrs_load_dataset()

    print(poma_concatenated_features_scaled_labels_df)
    print(updrs_concatenated_features_scaled_labels_df)

    # poma_concatenated_features_scaled_labels_df.to_csv('poma_features_scaled_labels_df_100_210823_00000000000000000000001.csv', index=False)
    # updrs_concatenated_features_scaled_labels_df.to_csv('updrs_features_scaled_labels_df_100_210823_00000000000000000000001.csv', index=False)

