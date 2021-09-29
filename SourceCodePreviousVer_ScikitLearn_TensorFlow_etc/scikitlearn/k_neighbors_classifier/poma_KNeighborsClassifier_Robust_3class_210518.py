import pickle

import joblib
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn_porter import Porter


poma_3class = pd.read_csv('../../../dataset/real_poma_3class_dataset_210518.csv')
# updrs_3class = pd.read_csv('../../dataset/real_updrs_3class_dataset_210518.csv')

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
rs_scaler.fit(poma_x_train)

# 훈련 데이터 스케일링
poma_x_train_scaled = rs_scaler.transform(poma_x_train_t)

# eval 데이터의 스케일링
poma_x_eval_scaled = rs_scaler.transform(poma_x_eval)

# 스케일링 된 결과 값으로 본래 값을 구할 수도 있다.
poma_x_origin = rs_scaler.inverse_transform(poma_x_train_scaled)

# AdaBoost 모델 생성
knn_clf = KNeighborsClassifier()

# 학습 (grid-search 전)
knn_clf.fit(poma_x_train_scaled, poma_y_train_t)

# 예측 (grid-search 전)
poma_pred_knn_before = knn_clf.predict(poma_x_eval_scaled)

print('grid search 전 KNN 예측 정확도 실제 y_eval / 예측 y_pred: ', accuracy_score(poma_y_eval, poma_pred_knn_before))

print('------------------ Before Eval KNN ------------------------')
print(confusion_matrix(poma_y_eval, poma_pred_knn_before))
print(classification_report(poma_y_eval, poma_pred_knn_before, target_names=['class 0', 'class 1', 'class 2']))

# 파라미터를 딕셔너리 형태로 설정 --> 이때 파라미터는 모델의 파라미터와 같아야 한다.
parameters = {'n_neighbors': [3, 5, 7, 9, 11, 13], 'leaf_size': [10, 30, 50, 70, 90, 110], 'p': [1, 2]}

grid_knn = GridSearchCV(knn_clf, param_grid=parameters, cv=5, refit=True)

# 학습 (grid-search 후)
grid_knn.fit(poma_x_train_scaled, poma_y_train_t)

# GridSearchCV 결과를 추출해 데이터 프레임으로 반환
scores_knn = pd.DataFrame(grid_knn.cv_results_)
scores_knn = scores_knn[['params', 'mean_test_score', 'rank_test_score',
                         'split0_test_score', 'split1_test_score', 'split2_test_score']]

print('KNN GridSearch 최적 파라미터: ', grid_knn.best_params_)
print('KNN Trees GridSearch 최고 점수: ', grid_knn.best_score_)

knn_estimator = grid_knn.best_estimator_

# GridSearchCV의 best_estmator_ 는 이미 최적 학습이 됐으므로 별도 학습이 필요 없음
poma_pred_knn_after = knn_estimator.predict(poma_x_eval_scaled)
print('grid-search 후 KNN eval 데이터세트 정확도: {0: .4f}'.format(accuracy_score(poma_y_eval, poma_pred_knn_after)))

print('------------------ After Eval KNN ------------------------')
print(confusion_matrix(poma_y_eval, poma_pred_knn_after))
print(classification_report(poma_y_eval, poma_pred_knn_after, target_names=['class 0', 'class 1', 'class 2']))

# 이제 전혀 사용하지 않은 테스트 데이터를 예측

# 테스트 데이터도 스케일링
poma_x_test_scaled = rs_scaler.transform(poma_x_test)

# 베스트 파라미터로 학습된 모델로 테스트 데이터 예측
poma_pred_knn_test = knn_estimator.predict(poma_x_test_scaled)
print('grid-search 후 KNN 테스트 데이터세트 정확도: {0: .4f}'.format(accuracy_score(poma_y_test, poma_pred_knn_test)))

print('------------------ Test KNN ------------------------')
print(confusion_matrix(poma_y_test, poma_pred_knn_test))
print(classification_report(poma_y_test, poma_pred_knn_test, target_names=['class 0', 'class 1', 'class 2']))


# grid search 전 KNN 예측 정확도 실제 y_eval / 예측 y_pred:  0.9301958307012003
# ------------------ Before Eval KNN ------------------------
# [[1497   53   40]
#  [  55 1010    7]
#  [  47   19  438]]
#               precision    recall  f1-score   support
#
#      class 0       0.94      0.94      0.94      1590
#      class 1       0.93      0.94      0.94      1072
#      class 2       0.90      0.87      0.89       504
#
#     accuracy                           0.93      3166
#    macro avg       0.92      0.92      0.92      3166
# weighted avg       0.93      0.93      0.93      3166

# KNN GridSearch 최적 파라미터:  {'leaf_size': 10, 'n_neighbors': 3, 'p': 1}
# KNN Trees GridSearch 최고 점수:  0.9392668591339968

# grid-search 후 KNN eval 데이터세트 정확도:  0.9425
# ------------------ After Eval KNN ------------------------
# [[1505   47   38]
#  [  35 1028    9]
#  [  43   10  451]]
#               precision    recall  f1-score   support
#
#      class 0       0.95      0.95      0.95      1590
#      class 1       0.95      0.96      0.95      1072
#      class 2       0.91      0.89      0.90       504
#
#     accuracy                           0.94      3166
#    macro avg       0.93      0.93      0.93      3166
# weighted avg       0.94      0.94      0.94      3166
#

# grid-search 후 KNN 테스트 데이터세트 정확도:  0.9411
# ------------------ Test KNN ------------------------
# [[1874   73   41]
#  [  47 1280   13]
#  [  42   17  571]]
#               precision    recall  f1-score   support
#
#      class 0       0.95      0.94      0.95      1988
#      class 1       0.93      0.96      0.94      1340
#      class 2       0.91      0.91      0.91       630
#
#     accuracy                           0.94      3958
#    macro avg       0.93      0.93      0.93      3958
# weighted avg       0.94      0.94      0.94      3958
#

# ---------------------------------------------------------------------------------------------------------------

# Save Model
saved_model = pickle.dumps(knn_estimator)

# Load the pickled model
clf_from_pickle = pickle.loads(saved_model)

# Use the loaded pickled model to make predictions (test data)
result_pickle = clf_from_pickle.predict(poma_x_test_scaled)

print('KNN Saved Model POMA Accuracy: {0: .4f}'.format(accuracy_score(poma_y_test, result_pickle)))

print('------------------ Test KNN Pickle Model ------------------------')
print(confusion_matrix(poma_y_test, result_pickle))
print(classification_report(poma_y_test, result_pickle, target_names=['class 0', 'class 1', 'class 2']))

# pickled model로 변환한 모델을 joblib 객체로 저장
joblib.dump(knn_estimator, 'knn_poma_3class_210615.pkl')

# ---------------------------------------------------------------------------------------------------------------

# # Porter 변환
# porter = Porter(knn_estimator, language='java')
# output = porter.export(embed_data=True)
# # print(output)
#
# # Porter 변환된 결과 파일 저장
# f = open("poma_KNeighborsClassifier_Robust_3class_210518.java", 'w')
# f.write(output)
# f.close()
#
# # 항상 원본 추정기와 변환 된 추정기 간의 무결성을 확인하고 계산해야 합니다.
# integrity = porter.integrity_score(poma_x_test_scaled)
# print(integrity)
# # code too large...

