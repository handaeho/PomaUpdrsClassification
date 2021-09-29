import pandas as pd
import numpy as np
import re

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
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

# nu SVC 모델 생성
nu_svc_clf = NuSVC(nu=0.3, kernel='rbf', gamma='auto', random_state=210518)

# 학습 (grid-search 전)
nu_svc_clf.fit(updrs_x_train_scaled, updrs_y_train)

# 예측 (grid-search 전)
updrs_pred_nu_before = nu_svc_clf.predict(updrs_x_test_scaled)

print('grid search 전 Nu 예측 정확도 실제 y_test / 예측 pred: ', accuracy_score(updrs_y_test, updrs_pred_nu_before))

print('----------------- Before test NU ---------------------')
print(confusion_matrix(updrs_y_test, updrs_pred_nu_before))
print(classification_report(updrs_y_test, updrs_pred_nu_before, target_names=['class 0', 'class 1']))

# 파라미터를 딕셔너리 형태로 설정 --> 이때 파라미터는 모델의 파라미터와 같아야 한다.
# nu_parameters = {'nu': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'gamma': ['scale', 'auto']}
nu_parameters = {'nu': [0.3, 0.4, 0.5, 0.6, 0.7]}

# grid-search
grid_nu_svc = GridSearchCV(nu_svc_clf, param_grid=nu_parameters, cv=5, refit=True)

# 학습 (grid-search 후)
grid_nu_svc.fit(updrs_x_train_scaled, updrs_y_train)

# GridSearchCV 결과를 추출해 데이터 프레임으로 반환
scores_df_nu = pd.DataFrame(grid_nu_svc.cv_results_)
scores_df_nu = scores_df_nu[['params', 'mean_test_score', 'rank_test_score',
                             'split0_test_score', 'split1_test_score', 'split2_test_score']]

# print(scores_df_nu)

print('Nu GridSearch 최적 파라미터: ', grid_nu_svc.best_params_)
print('Nu GridSearch 최고 점수: ', grid_nu_svc.best_score_)

nu_estimator = grid_nu_svc.best_estimator_

# GridSearchCV의 best_estmator_ 는 이미 최적 학습이 됐으므로 별도 학습이 필요 없음

updrs_pred_nu_after = nu_estimator.predict(updrs_x_test_scaled)
print('grid-search 후 Nu test 데이터세트 정확도: {0: .4f}'.format(accuracy_score(updrs_y_test, updrs_pred_nu_after)))

print('----------------- After test NU ---------------------')
print(confusion_matrix(updrs_y_test, updrs_pred_nu_after))
# print(classification_report(updrs_y_test, updrs_pred_nu_after, target_names=['class 0', 'class 1']))

# report = classification_report(updrs_y_test, updrs_pred_nu_after, target_names=['class 0', 'class 1'], output_dict=True)
# df_classification_report = pd.DataFrame(report).transpose()
# print(df_classification_report)

c_matrix = confusion_matrix(updrs_y_test, updrs_pred_nu_after)
df_c_matrix = pd.DataFrame(c_matrix)
print('---------------------')
print('민감도(sensitivity) TP / TP + FN: ', df_c_matrix.iloc[0, 0] / (df_c_matrix.iloc[0, 0] + df_c_matrix.iloc[1, 0]))
print('특이도(specificity) TN / FP + TN: ', df_c_matrix.iloc[1, 1] / (df_c_matrix.iloc[1, 1] + df_c_matrix.iloc[0, 1]))

# grid search 전 Nu 예측 정확도 실제 y_test / 예측 pred:  0.8633333333333333
# ----------------- Before test NU ---------------------
# [[111  27]
#  [ 14 148]]
#               precision    recall  f1-score   support
#
#      class 0       0.89      0.80      0.84       138
#      class 1       0.85      0.91      0.88       162
#
#     accuracy                           0.86       300
#    macro avg       0.87      0.86      0.86       300
# weighted avg       0.87      0.86      0.86       300
#

# Nu GridSearch 최적 파라미터:  {'nu': 0.3}
# Nu GridSearch 최고 점수:  0.875

# grid-search 후 Nu test 데이터세트 정확도:  0.8633
# ----------------- After test NU ---------------------
# [[111  27]
#  [ 14 148]]
#               precision    recall  f1-score   support
#
#      class 0       0.89      0.80      0.84       138
#      class 1       0.85      0.91      0.88       162
#
#     accuracy                           0.86       300
#    macro avg       0.87      0.86      0.86       300
# weighted avg       0.87      0.86      0.86       300

