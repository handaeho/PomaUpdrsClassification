import pandas as pd
import pickle

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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

# Decision Tree 모델 생성
rf_clf = RandomForestClassifier(random_state=210518)

# 학습 (grid-search 전)
rf_clf.fit(updrs_x_train_scaled, updrs_y_train_t)

# 예측 (grid-search 전)
updrs_pred_rf_before = rf_clf.predict(updrs_x_eval_scaled)

print('grid search 전 RF 예측 정확도 실제 updrs_y_eval / 예측 updrs_y_pred: ', accuracy_score(updrs_y_eval, updrs_pred_rf_before))

print('------------------ Before Eval RF ------------------------')
print(confusion_matrix(updrs_y_eval, updrs_pred_rf_before))
print(classification_report(updrs_y_eval, updrs_pred_rf_before, target_names=['class 0', 'class 1', 'class 2']))

# 파라미터를 딕셔너리 형태로 설정 --> 이때 파라미터는 모델의 파라미터와 같아야 한다.

# parameters = {'n_estimators': [100, 200, 300], 'criterion': ['gini', 'entropy'], 'max_depth': [1, 2, 3, 4, 5],
#               'min_samples_split': [2, 3, 4, 5, 6], 'min_samples_leaf': [1, 2, 3, 4, 5],
#               'max_features': [2, 3.1, 'auto', 'sqrt', 'log2', None], 'max_leaf_nodes': [None, 1, 2, 3, 4],
#               'min_impurity_decrease': [0, 1.0, 2.0, 3.0, 4.0, 5.0], 'bootstrap': [True, False],
#               'oob_score': [True, False], 'n_jobs': [None, 1, 2, 3, 4, 5], 'warm_start': [False, True]}
# --> 아런 식으로 설정하면 된다.

parameters = {'n_estimators': [100, 200, 300, 400, 500], 'min_samples_split': [2, 3, 4, 5, 6, 7, 8]}

grid_rf = GridSearchCV(rf_clf, param_grid=parameters, cv=5, refit=True)

# 학습 (grid-search 후)
grid_rf.fit(updrs_x_train_scaled, updrs_y_train_t)

# GridSearchCV 결과를 추출해 데이터 프레임으로 반환
scores_rf = pd.DataFrame(grid_rf.cv_results_)
scores_tf = scores_rf[['params', 'mean_test_score', 'rank_test_score',
                       'split0_test_score', 'split1_test_score', 'split2_test_score']]

# print(scores_dtree)

print('Random Forest GridSearch 최적 파라미터: ', grid_rf.best_params_)
print('Random Forest GridSearch 최고 점수: ', grid_rf.best_score_)

rf_estimator = grid_rf.best_estimator_

# GridSearchCV의 best_estmator_ 는 이미 최적 학습이 됐으므로 별도 학습이 필요 없음
pred_rf_after = rf_estimator.predict(updrs_x_eval_scaled)
print('grid-search 후 RF 검증 데이터세트 정확도: {0: .4f}'.format(accuracy_score(updrs_y_eval, pred_rf_after)))

print('------------------ After Eval RF ------------------------')
print(confusion_matrix(updrs_y_eval, pred_rf_after))
print(classification_report(updrs_y_eval, pred_rf_after, target_names=['class 0', 'class 1', 'class 2']))

# 이제 전혀 사용하지 않은 테스트 데이터를 예측

# 테스트 데이터도 스케일링
updrs_x_test_scaled = rs_scaler.transform(updrs_x_test)

# 베스트 파라미터로 학습된 모델로 테스트 데이터 예측
pred_rf_test = rf_estimator.predict(updrs_x_test_scaled)
print('grid-search 후 RF 테스트 데이터세트 정확도: {0: .4f}'.format(accuracy_score(updrs_y_test, pred_rf_test)))

print('------------------ Test RF ------------------------')
print(confusion_matrix(updrs_y_test, pred_rf_test))
print(classification_report(updrs_y_test, pred_rf_test, target_names=['class 0', 'class 1', 'class 2']))


# grid search 전 RF 예측 정확도 실제 updrs_y_eval / 예측 updrs_y_pred:  0.917877447883765
# ------------------ Before Eval RF ------------------------
# [[1368   77    5]
#  [  86 1115   21]
#  [  30   41  423]]
#               precision    recall  f1-score   support
#
#      class 0       0.92      0.94      0.93      1450
#      class 1       0.90      0.91      0.91      1222
#      class 2       0.94      0.86      0.90       494
#
#     accuracy                           0.92      3166
#    macro avg       0.92      0.90      0.91      3166
# weighted avg       0.92      0.92      0.92      3166

# Random Forest GridSearch 최적 파라미터:  {'min_samples_split': 2, 'n_estimators': 500}
# Random Forest GridSearch 최고 점수:  0.9140732535897402

# grid-search 후 RF 검증 데이터세트 정확도:  0.9239
# ------------------ After Eval RF ------------------------
# [[1371   74    5]
#  [  82 1121   19]
#  [  26   35  433]]
#               precision    recall  f1-score   support
#
#      class 0       0.93      0.95      0.94      1450
#      class 1       0.91      0.92      0.91      1222
#      class 2       0.95      0.88      0.91       494
#
#     accuracy                           0.92      3166
#    macro avg       0.93      0.91      0.92      3166
# weighted avg       0.92      0.92      0.92      3166
#

# grid-search 후 RF 테스트 데이터세트 정확도:  0.9275
# ------------------ Test RF ------------------------
# [[1718   90    4]
#  [  71 1426   31]
#  [  34   57  527]]
#               precision    recall  f1-score   support
#
#      class 0       0.94      0.95      0.95      1812
#      class 1       0.91      0.93      0.92      1528
#      class 2       0.94      0.85      0.89       618
#
#     accuracy                           0.93      3958
#    macro avg       0.93      0.91      0.92      3958
# weighted avg       0.93      0.93      0.93      3958

# ---------------------------------------------------------------------------------------------------------------

# Save Model
saved_model = pickle.dumps(rf_estimator)

# Load the pickled model
clf_from_pickle = pickle.loads(saved_model)

# Use the loaded pickled model to make predictions (test data)
result_pickle = clf_from_pickle.predict(updrs_x_test_scaled)

print('RF Saved Model UPDRS Accuracy: {0: .4f}'.format(accuracy_score(updrs_y_test, result_pickle)))

print('------------------ Test RF Pickle Model ------------------------')
print(confusion_matrix(updrs_y_test, result_pickle))
print(classification_report(updrs_y_test, result_pickle, target_names=['class 0', 'class 1', 'class 2']))

# pickled model로 변환한 모델을 joblib 객체로 저장
joblib.dump(rf_estimator, 'rf_updrs_3class_210615.pkl')

# ---------------------------------------------------------------------------------------------------------------

# # Porter 변환
# porter = Porter(rf_estimator, language='java')
# output = porter.export(embed_data=True)
# # print(output)
#
# # Porter 변환된 결과 파일 저장
# f = open("updrs_RandomForest_Robust_3class_210518.java", 'w')
# f.write(output)
# f.close()
#
# # 항상 원본 추정기와 변환 된 추정기 간의 무결성을 확인하고 계산해야 합니다.
# integrity = porter.integrity_score(updrs_x_test_scaled)
# print(integrity)
