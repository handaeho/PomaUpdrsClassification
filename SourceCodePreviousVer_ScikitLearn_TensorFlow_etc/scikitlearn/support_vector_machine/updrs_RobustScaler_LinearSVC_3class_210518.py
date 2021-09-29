import pandas as pd
import numpy as np
import re

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import LinearSVC, NuSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn_porter import Porter


# poma_3class = pd.read_csv('../../dataset/real_poma_3class_dataset_210518.csv')
updrs_3class = pd.read_csv('../../../dataset/real_updrs_3class_dataset_210518.csv')

# poma_dataset_3class = poma_3class.copy()
updrs_dataset_3class = updrs_3class.copy()

# print(poma_dataset_3class)
print(updrs_dataset_3class)

# poma_features = poma_dataset_3class
# poma_labels = poma_dataset_3class.pop('poma_danger_3class')

updrs_features = updrs_3class
updrs_labels = updrs_3class.pop('updrs_danger_3class')

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

print(updrs_x_train.shape, updrs_y_train.shape)       # (15828, 95) (15828,)
print(updrs_x_train_t.shape, updrs_y_train_t.shape)   # (12662, 95) (12662,)
print(updrs_x_eval.shape, updrs_y_eval.shape)         # (3166, 95) (3166,)
print(updrs_x_test.shape, updrs_y_test.shape)         # (3958, 95) (3958,)

# 변형 객체 생성
rs_scaler = RobustScaler()

# 훈련데이터의 모수 분포 저장
rs_scaler.fit(updrs_x_train_t)

# 훈련 데이터 스케일링
updrs_x_train_scaled = rs_scaler.transform(updrs_x_train_t)

# 검증 데이터의 스케일링
updrs_x_eval_scaled = rs_scaler.transform(updrs_x_eval)

# 스케일링 된 결과 값으로 본래 값을 구할 수도 있다.
updrs_x_origin = rs_scaler.inverse_transform(updrs_x_train_scaled)

# linear, nu SVC 모델 생성
linear_svc_clf = LinearSVC(C=10.0, random_state=210518)

# 학습 (grid-search 전)
linear_svc_clf.fit(updrs_x_train_scaled, updrs_y_train_t)

# 예측 (grid-search 전)
updrs_pred_linear_before = linear_svc_clf.predict(updrs_x_eval_scaled)

print('grid search 전 Linear 예측 정확도 실제 y_eval / 예측 pred: ', accuracy_score(updrs_y_eval, updrs_pred_linear_before))

print('------------------ Before Linear ------------------------')
print(confusion_matrix(updrs_y_eval, updrs_pred_linear_before))
print(classification_report(updrs_y_eval, updrs_pred_linear_before, target_names=['class 0', 'class 1', 'class 2']))

# 파라미터를 딕셔너리 형태로 설정 --> 이때 파라미터는 모델의 파라미터와 같아야 한다.
# linear_parameters = {'penalty': ['l2', 'l1'], 'loss': ['squared_hinge', 'hidge'], 'tol': [0.0001, 0.0005, 0.001],
#                      'C': [1.0, 2.0, 3.0, 4.0, 5.0], 'max_iter': [1000, 2000, 3000, 4000, 5000]}
linear_parameters = {'penalty': ['l2', 'l1'], 'loss': ['squared_hinge', 'hidge'], 'C': [1.0, 3.0, 5.0, 7.0, 10.0]}

# grid-search
grid_linear_svc = GridSearchCV(linear_svc_clf, param_grid=linear_parameters, cv=5, refit=True)

# 학습 (grid-search 후)
grid_linear_svc.fit(updrs_x_train_scaled, updrs_y_train_t)

# GridSearchCV 결과를 추출해 데이터 프레임으로 반환
scores_df_linear = pd.DataFrame(grid_linear_svc.cv_results_)
scores_df_linear = scores_df_linear[['params', 'mean_test_score', 'rank_test_score',
                                     'split0_test_score', 'split1_test_score', 'split2_test_score']]

# print(scores_df_linear)

print('Linear GridSearch 최적 파라미터: ', grid_linear_svc.best_params_)
print('Linear GridSearch 최고 점수: ', grid_linear_svc.best_score_)

linear_estimator = grid_linear_svc.best_estimator_

# GridSearchCV의 best_estmator_ 는 이미 최적 학습이 됐으므로 별도 학습이 필요 없음
updrs_pred_linear_after = linear_estimator.predict(updrs_x_eval_scaled)
print('grid-search 후 Linear 검증 데이터세트 정확도: {0: .4f}'.format(accuracy_score(updrs_y_eval, updrs_pred_linear_after)))

print('------------------ After Linear ------------------------')
print(confusion_matrix(updrs_y_eval, updrs_pred_linear_after))
print(classification_report(updrs_y_eval, updrs_pred_linear_after, target_names=['class 0', 'class 1', 'class 2']))

# 이제 전혀 사용하지 않은 테스트 데이터를 예측

# 테스트 데이터도 스케일링
updrs_x_test_scaled = rs_scaler.transform(updrs_x_test)

# 베스트 파라미터로 학습된 모델로 테스트 데이터 예측
updrs_pred_linear_test = linear_estimator.predict(updrs_x_test_scaled)
print('grid-search 후 Linear 테스트 데이터세트 정확도: {0: .4f}'.format(accuracy_score(updrs_y_test, updrs_pred_linear_test)))

print('------------------ Test Linear ------------------------')
print(confusion_matrix(updrs_y_test, updrs_pred_linear_test))
print(classification_report(updrs_y_test, updrs_pred_linear_test, target_names=['class 0', 'class 1', 'class 2']))


# grid search 전 Linear 예측 정확도 실제 y_eval / 예측 pred:  0.613076437144662
# ------------------ Before Linear ------------------------
# [[1005  380   65]
#  [ 410  719   93]
#  [  92  185  217]]
#               precision    recall  f1-score   support
#
#      class 0       0.67      0.69      0.68      1450
#      class 1       0.56      0.59      0.57      1222
#      class 2       0.58      0.44      0.50       494
#
#     accuracy                           0.61      3166
#    macro avg       0.60      0.57      0.58      3166
# weighted avg       0.61      0.61      0.61      3166

# Linear GridSearch 최적 파라미터:  {'C': 3.0, 'loss': 'squared_hinge', 'penalty': 'l2'}
# Linear GridSearch 최고 점수:  0.6651399005481514

# grid-search 후 Linear 검증 데이터세트 정확도:  0.6475
# ------------------ After Linear ------------------------
# [[1056  356   38]
#  [ 388  737   97]
#  [ 100  137  257]]
#               precision    recall  f1-score   support
#
#      class 0       0.68      0.73      0.71      1450
#      class 1       0.60      0.60      0.60      1222
#      class 2       0.66      0.52      0.58       494
#
#     accuracy                           0.65      3166
#    macro avg       0.65      0.62      0.63      3166
# weighted avg       0.65      0.65      0.65      3166
#

# grid-search 후 Linear 테스트 데이터세트 정확도:  0.6478
# ------------------ Test Linear ------------------------
# [[1338  425   49]
#  [ 475  922  131]
#  [ 143  171  304]]
#               precision    recall  f1-score   support
#
#      class 0       0.68      0.74      0.71      1812
#      class 1       0.61      0.60      0.61      1528
#      class 2       0.63      0.49      0.55       618
#
#     accuracy                           0.65      3958
#    macro avg       0.64      0.61      0.62      3958
# weighted avg       0.65      0.65      0.64      3958
#


# Porter 변환
porter_linear = Porter(linear_estimator, language='java')

output_linear = porter_linear.export(embed_data=True)

# Porter 변환된 결과 파일 저장
f_linear = open("updrs_RobustScalerSVC_3class_linear_210518.java", 'w')

f_linear.write(output_linear)

f_linear.close()

# 항상 원본 추정기와 변환 된 추정기 간의 무결성을 확인하고 계산해야 합니다.
integrity_linear = porter_linear.integrity_score(updrs_x_test_scaled)
print(integrity_linear)


