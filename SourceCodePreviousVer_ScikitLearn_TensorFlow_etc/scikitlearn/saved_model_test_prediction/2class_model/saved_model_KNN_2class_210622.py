import pandas as pd
import re

from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import RobustScaler


def poma_load_dataset():
    poma_2class = pd.read_csv('../../../../dataset/real_poma_2class_dataset_210518.csv')

    poma_dataset_2class = poma_2class.copy()

    poma_dataset_2class = poma_dataset_2class.sample(n=1500)

    # TF2 Pipe-line에는 컬럼명에 특수문자 불가.
    cols = [re.sub(r'[\W_]', "", i) for i in poma_dataset_2class.columns]

    for i in range(len(cols)):
        cols[i] = cols[i] + str(i)

    poma_dataset_2class.columns = cols

    # print(poma_dataset_3class)

    # 'danger' 컬럼 -> 라벨
    features = poma_dataset_2class.iloc[:, 1:].values
    labels = poma_dataset_2class['pomadanger2class0'].values

    # 변형 객체 생성 / 모수 분포 저장 (fit)
    rs_scaler = RobustScaler().fit(features)

    # 데이터 스케일링 (transform)
    features_scaled = rs_scaler.transform(features)

    return features_scaled, labels


def updrs_load_dataset():
    updrs_2class = pd.read_csv('../../../../dataset/real_updrs_2class_dataset_210518.csv')

    updrs_dataset_2class_all = updrs_2class.copy()

    updrs_dataset_2class = updrs_dataset_2class_all.sample(n=1500)

    # TF2 Pipe-line에는 컬럼명에 특수문자 불가.
    cols = [re.sub(r'[\W_]', "", i) for i in updrs_dataset_2class.columns]

    for i in range(len(cols)):
        cols[i] = cols[i] + str(i)

    updrs_dataset_2class.columns = cols

    # print(updrs_dataset_3class)

    # 'danger' 컬럼 -> 라벨
    features = updrs_dataset_2class.iloc[:, 1:].values
    labels = updrs_dataset_2class['updrsdanger2class0'].values

    # 변형 객체 생성
    rs_scaler = RobustScaler().fit(features)

    # 데이터 스케일링
    features_scaled = rs_scaler.transform(features)

    return features_scaled, labels


def poma_predict_knn(dataset):
    # load saved model
    clf_from_joblib = joblib.load('../../k_neighbors_classifier/knn_poma_2class_210622.pkl')

    # prediction
    result = clf_from_joblib.predict(dataset)

    return result


def updrs_predict_knn(dataset):
    # load saved model
    clf_from_joblib = joblib.load('../../k_neighbors_classifier/knn_updrs_2class_210622.pkl')

    # prediction
    result = clf_from_joblib.predict(dataset)

    return result


if __name__ == '__main__':
    poma_x, poma_y = poma_load_dataset()
    # print(poma_x)
    # print(poma_y)

    updrs_x, updrs_y = updrs_load_dataset()
    # print(updrs_x)
    # print(updrs_y)

    result_poma_predict = poma_predict_knn(poma_x)

    result_updrs_predict = updrs_predict_knn(updrs_x)

    print('KNN Saved Model POMA Accuracy: {0: .4f}'.format(accuracy_score(poma_y, result_poma_predict)))

    print('------------------ Test KNN POMA  ------------------------')
    print(confusion_matrix(poma_y, result_poma_predict))
    print(classification_report(poma_y, result_poma_predict, target_names=['class 0', 'class 1']))

    print('KNN Saved Model UPDRS Accuracy: {0: .4f}'.format(accuracy_score(updrs_y, result_updrs_predict)))

    print('------------------ Test KNN UPDRS ------------------------')
    print(confusion_matrix(updrs_y, result_updrs_predict))
    print(classification_report(updrs_y, result_updrs_predict, target_names=['class 0', 'class 1']))