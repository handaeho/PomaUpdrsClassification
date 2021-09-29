import pandas as pd
import numpy as np
import re

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import LinearSVC, NuSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn_porter import Porter


# poma_2class = pd.read_csv('../../dataset/poma_dataset_210518_1500.csv')
updrs_2class = pd.read_csv('../../../dataset/updrs_dataset_210518_1500.csv')

# poma_dataset_2class = poma_2class.copy()
updrs_dataset_2class = updrs_2class.copy()

# print(poma_dataset_2class)
print(updrs_dataset_2class)

# poma_features = poma_dataset_2class
# poma_labels = poma_dataset_2class.pop('poma_danger_2class')

updrs_features = updrs_dataset_2class
updrs_labels = updrs_dataset_2class.pop('updrs_danger_2class')

updrs_x_train, updrs_x_test, updrs_y_train, updrs_y_test = train_test_split(updrs_features, updrs_labels,
                                                                            test_size=0.2,
                                                                            stratify=updrs_labels,
                                                                            shuffle=True,
                                                                            random_state=1234)

print(updrs_x_train.shape, updrs_y_train.shape)       # (15828, 95) (15828,)
print(updrs_x_test.shape, updrs_y_test.shape)         # (3958, 95) (3958,)


# 변형 객체 생성
rs_scaler = RobustScaler()

# 훈련데이터의 모수 분포 저장
rs_scaler.fit(updrs_x_train)

# 훈련 데이터 스케일링
updrs_x_train_scaled = rs_scaler.transform(updrs_x_train)

# test 데이터의 스케일링
updrs_x_test_scaled = rs_scaler.transform(updrs_x_test)

# 스케일링 된 결과 값으로 본래 값을 구할 수도 있다.
updrs_x_origin = rs_scaler.inverse_transform(updrs_x_train_scaled)

# linear, nu SVC 모델 생성
linear_svc_clf = LinearSVC(C=10.0, random_state=210518)

# 학습 (grid-search 전)
linear_svc_clf.fit(updrs_x_train_scaled, updrs_y_train)

# 예측 (grid-search 전)
updrs_pred_linear_before = linear_svc_clf.predict(updrs_x_test_scaled)

print('grid search 전 Linear 예측 정확도 실제 y_test / 예측 pred: ', accuracy_score(updrs_y_test, updrs_pred_linear_before))

print('------------------ Before test Linear ------------------------')
print(confusion_matrix(updrs_y_test, updrs_pred_linear_before))
print(classification_report(updrs_y_test, updrs_pred_linear_before, target_names=['class 0', 'class 1']))

# 파라미터를 딕셔너리 형태로 설정 --> 이때 파라미터는 모델의 파라미터와 같아야 한다.
# linear_parameters = {'penalty': ['l2', 'l1'], 'loss': ['squared_hinge', 'hidge'], 'tol': [0.0001, 0.0005, 0.001],
#                      'C': [1.0, 2.0, 3.0, 4.0, 5.0], 'max_iter': [1000, 2000, 3000, 4000, 5000]}
linear_parameters = {'penalty': ['l2', 'l1'], 'loss': ['squared_hinge', 'hidge'], 'C': [1.0, 3.0, 5.0, 7.0, 10.0]}

# grid-search
grid_linear_svc = GridSearchCV(linear_svc_clf, param_grid=linear_parameters, cv=5, refit=True)

# 학습 (grid-search 후)
grid_linear_svc.fit(updrs_x_train_scaled, updrs_y_train)

# GridSearchCV 결과를 추출해 데이터 프레임으로 반환
scores_df_linear = pd.DataFrame(grid_linear_svc.cv_results_)
scores_df_linear = scores_df_linear[['params', 'mean_test_score', 'rank_test_score',
                                     'split0_test_score', 'split1_test_score', 'split2_test_score']]

# print(scores_df_linear)

print('Linear GridSearch 최적 파라미터: ', grid_linear_svc.best_params_)
print('Linear GridSearch 최고 점수: ', grid_linear_svc.best_score_)

linear_estimator = grid_linear_svc.best_estimator_

# GridSearchCV의 best_estmator_ 는 이미 최적 학습이 됐으므로 별도 학습이 필요 없음
updrs_pred_linear_after = linear_estimator.predict(updrs_x_test_scaled)
print('grid-search 후 Linear test 데이터세트 정확도: {0: .4f}'.format(accuracy_score(updrs_y_test, updrs_pred_linear_after)))

print('------------------ After test Linear ------------------------')
print(confusion_matrix(updrs_y_test, updrs_pred_linear_after))
print(classification_report(updrs_y_test, updrs_pred_linear_after, target_names=['class 0', 'class 1']))

# grid search 전 Linear 예측 정확도 실제 y_test / 예측 pred:  0.6966666666666667
# ------------------ Before test Linear ------------------------
# [[ 88  50]
#  [ 41 121]]
#               precision    recall  f1-score   support
#
#      class 0       0.68      0.64      0.66       138
#      class 1       0.71      0.75      0.73       162
#
#     accuracy                           0.70       300
#    macro avg       0.69      0.69      0.69       300
# weighted avg       0.70      0.70      0.70       300

# Linear GridSearch 최적 파라미터:  {'C': 7.0, 'loss': 'squared_hinge', 'penalty': 'l2'}
# Linear GridSearch 최고 점수:  0.7275

# grid-search 후 Linear test 데이터세트 정확도:  0.7400
# ------------------ After test Linear ------------------------
# [[100  38]
#  [ 40 122]]
#               precision    recall  f1-score   support
#
#      class 0       0.71      0.72      0.72       138
#      class 1       0.76      0.75      0.76       162
#
#     accuracy                           0.74       300
#    macro avg       0.74      0.74      0.74       300
# weighted avg       0.74      0.74      0.74       300

