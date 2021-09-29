import pickle

import joblib
import pandas as pd
import numpy as np
import re

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn_porter import Porter


poma_3class = pd.read_csv('../../../dataset/real_poma_3class_dataset_210518.csv')
# updrs_3class = pd.read_csv('../../dataset/real_updrs_3class_dataset_210518.csv.csv')

poma_dataset_3class = poma_3class.copy()
# updrs_dataset_3class = updrs_3class.copy()

print(poma_dataset_3class)
# print(updrs_dataset_3class)

poma_features = poma_dataset_3class
poma_labels = poma_dataset_3class.pop('poma_danger_3class')

# updrs_features = updrs_dataset
# updrs_labels = updrs_dataset.pop('updrs_danger_3class')

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

print(poma_x_train.shape, poma_y_train.shape)       # (15828, 95) (15828,)
print(poma_x_train_t.shape, poma_y_train_t.shape)   # (12662, 95) (12662,)
print(poma_x_eval.shape, poma_y_eval.shape)         # (3166, 95) (3166,)
print(poma_x_test.shape, poma_y_test.shape)         # (3958, 95) (3958,)

# 변형 객체 생성
rs_scaler = RobustScaler()

# 훈련데이터의 모수 분포 저장
rs_scaler.fit(poma_x_train_t)

# 훈련 데이터 스케일링
poma_x_train_scaled = rs_scaler.transform(poma_x_train_t)

# eval 데이터의 스케일링
poma_x_eval_scaled = rs_scaler.transform(poma_x_eval)

# 스케일링 된 결과 값으로 본래 값을 구할 수도 있다.
poma_x_origin = rs_scaler.inverse_transform(poma_x_train_scaled)

# Decision Tree 모델 생성
dtree_clf = DecisionTreeClassifier(random_state=210518)

# 학습 (grid-search 전)
dtree_clf.fit(poma_x_train_scaled, poma_y_train_t)

# 예측 (grid-search 전)
poma_pred_dtree_before = dtree_clf.predict(poma_x_eval_scaled)

print('grid search 전 DT 예측 정확도 실제 y_eval / 예측 y_pred: ', accuracy_score(poma_y_eval, poma_pred_dtree_before))

print('------------------ Before Eval d-Tree ------------------------')
print(confusion_matrix(poma_y_eval, poma_pred_dtree_before))
print(classification_report(poma_y_eval, poma_pred_dtree_before, target_names=['class 0', 'class 1', 'class 2']))


# 파라미터를 딕셔너리 형태로 설정 --> 이때 파라미터는 모델의 파라미터와 같아야 한다.
parameters = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'],
              'max_depth': [None, 1, 2, 3, 4, 5], 'min_samples_split': [2, 3, 4, 5, 6]}

grid_dtree = GridSearchCV(dtree_clf, param_grid=parameters, cv=5, refit=True)

# 학습 (grid-search 후)
grid_dtree.fit(poma_x_train_scaled, poma_y_train_t)

# GridSearchCV 결과를 추출해 데이터 프레임으로 반환
scores_dtree = pd.DataFrame(grid_dtree.cv_results_)
scores_dtree = scores_dtree[['params', 'mean_test_score', 'rank_test_score',
                             'split0_test_score', 'split1_test_score', 'split2_test_score']]

# print(scores_dtree)

print('Decision Tree GridSearch 최적 파라미터: ', grid_dtree.best_params_)
print('Decision Tree GridSearch 최고 점수: ', grid_dtree.best_score_)

dtree_estimator = grid_dtree.best_estimator_

# GridSearchCV의 best_estmator_ 는 이미 최적 학습이 됐으므로 별도 학습이 필요 없음
poma_pred_dtree_after = dtree_estimator.predict(poma_x_eval_scaled)

print('grid-search 후 DT eval 데이터세트 정확도: {0: .4f}'.format(accuracy_score(poma_y_eval, poma_pred_dtree_after)))

print('------------------ After Eval d-Tree ------------------------')
print(confusion_matrix(poma_y_eval, poma_pred_dtree_after))
print(classification_report(poma_y_eval, poma_pred_dtree_after, target_names=['class 0', 'class 1', 'class 2']))

# 이제 전혀 사용하지 않은 테스트 데이터를 예측

# 테스트 데이터도 스케일링
poma_x_test_scaled = rs_scaler.transform(poma_x_test)

# 베스트 파라미터로 학습된 모델로 테스트 데이터 예측
poma_pred_dt_test = dtree_estimator.predict(poma_x_test_scaled)
print('grid-search 후 DT 테스트 데이터세트 정확도: {0: .4f}'.format(accuracy_score(poma_y_test, poma_pred_dt_test)))

print('------------------ Test DT ------------------------')
print(confusion_matrix(poma_y_test, poma_pred_dt_test))
print(classification_report(poma_y_test, poma_pred_dt_test, target_names=['class 0', 'class 1', 'class 2']))


# grid search 전 DT 예측 정확도 실제 y_eval / 예측 y_pred:  0.817435249526216
# ------------------ Before Eval d-Tree ------------------------
# [[1356  148   86]
#  [ 157  866   49]
#  [  85   53  366]]
#               precision    recall  f1-score   support
#
#      class 0       0.85      0.85      0.85      1590
#      class 1       0.81      0.81      0.81      1072
#      class 2       0.73      0.73      0.73       504
#
#     accuracy                           0.82      3166
#    macro avg       0.80      0.80      0.80      3166
# weighted avg       0.82      0.82      0.82      3166

# Decision Tree GridSearch 최적 파라미터:  {'criterion': 'entropy', 'max_depth': None,
#                                       'min_samples_split': 3, 'splitter': 'best'}
# Decision Tree GridSearch 최고 점수:  0.8138530637293883

# grid-search 후 DT eval 데이터세트 정확도:  0.8282
# ------------------ After Eval d-Tree ------------------------
# [[1358  152   80]
#  [ 141  884   47]
#  [  86   38  380]]
#               precision    recall  f1-score   support
#
#      class 0       0.86      0.85      0.86      1590
#      class 1       0.82      0.82      0.82      1072
#      class 2       0.75      0.75      0.75       504
#
#     accuracy                           0.83      3166
#    macro avg       0.81      0.81      0.81      3166
# weighted avg       0.83      0.83      0.83      3166
#

# grid-search 후 DT 테스트 데이터세트 정확도:  0.8196
# ------------------ Test DT ------------------------
# [[1693  180  115]
#  [ 186 1081   73]
#  [  97   63  470]]
#               precision    recall  f1-score   support
#
#      class 0       0.86      0.85      0.85      1988
#      class 1       0.82      0.81      0.81      1340
#      class 2       0.71      0.75      0.73       630
#
#     accuracy                           0.82      3958
#    macro avg       0.80      0.80      0.80      3958
# weighted avg       0.82      0.82      0.82      3958
#

# ---------------------------------------------------------------------------------------------------------------

# Save Model
saved_model = pickle.dumps(dtree_estimator)

# Load the pickled model
clf_from_pickle = pickle.loads(saved_model)

# Use the loaded pickled model to make predictions (test data)
result_pickle = clf_from_pickle.predict(poma_x_test_scaled)

print('DT Saved Model POMA Accuracy: {0: .4f}'.format(accuracy_score(poma_y_test, result_pickle)))

print('------------------ Test DT Pickle Model ------------------------')
print(confusion_matrix(poma_y_test, result_pickle))
print(classification_report(poma_y_test, result_pickle, target_names=['class 0', 'class 1', 'class 2']))

# pickled model로 변환한 모델을 joblib 객체로 저장
joblib.dump(dtree_estimator, 'dt_poma_3class_210615.pkl')

# ---------------------------------------------------------------------------------------------------------------

# # Porter 변환
# porter = Porter(dtree_estimator, language='java')
# output = porter.export(embed_data=True)
# # print(output)
#
# # Porter 변환된 결과 파일 저장
# f = open("poma_RobustScalerDT_3class_210518.java", 'w')
# f.write(output)
# f.close()
#
# # 항상 원본 추정기와 변환 된 추정기 간의 무결성을 확인하고 계산해야 합니다.
# integrity = porter.integrity_score(poma_x_test_scaled)
# print(integrity)
# # => code too large...

