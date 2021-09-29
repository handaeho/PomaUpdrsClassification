import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler


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

# poma_x_train, poma_x_test, poma_y_train, poma_y_test = train_test_split(poma_features, poma_labels,
#                                                                         test_size=0.2,
#                                                                         stratify=poma_labels,
#                                                                         shuffle=True,
#                                                                         random_state=1234)
#
# poma_x_train_t, poma_x_eval, poma_y_train_t, poma_y_eval = train_test_split(poma_x_train, poma_y_train,
#                                                                             test_size=0.2,
#                                                                             stratify=poma_y_train,
#                                                                             shuffle=True,
#                                                                             random_state=1234)

updrs_x_train, updrs_x_test, updrs_y_train, updrs_y_test = train_test_split(updrs_features, updrs_labels,
                                                                            test_size=0.2,
                                                                            stratify=updrs_labels,
                                                                            shuffle=True,
                                                                            random_state=1234)

# updrs_x_train_t, updrs_x_eval, updrs_y_train_t, updrs_y_eval = train_test_split(updrs_x_train, updrs_y_train,
#                                                                                 test_size=0.2,
#                                                                                 stratify=updrs_y_train,
#                                                                                 shuffle=True,
#                                                                                 random_state=1234)

print(updrs_x_train.shape, updrs_y_train.shape)       # (15828, 95) (15828,)
# print(updrs_x_train_t.shape, updrs_y_train_t.shape)   # (12662, 95) (12662,)
# print(updrs_x_eval.shape, updrs_y_eval.shape)         # (3166, 95) (3166,)
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

# AdaBoost 모델 생성
knn_clf = KNeighborsClassifier()

# 학습 (grid-search 전)
knn_clf.fit(updrs_x_train_scaled, updrs_y_train)

# 예측 (grid-search 전)
updrs_pred_knn_before = knn_clf.predict(updrs_x_test_scaled)

print('grid search 전 KNN 예측 정확도 실제 y_test / 예측 y_pred: ', accuracy_score(updrs_y_test, updrs_pred_knn_before))

print('------------------ Before Test KNN ------------------------')
print(confusion_matrix(updrs_y_test, updrs_pred_knn_before))
print(classification_report(updrs_y_test, updrs_pred_knn_before, target_names=['class 0', 'class 1']))

# 파라미터를 딕셔너리 형태로 설정 --> 이때 파라미터는 모델의 파라미터와 같아야 한다.
parameters = {'n_neighbors': [3, 5, 7, 9, 11, 13], 'leaf_size': [10, 30, 50, 70, 90, 110], 'p': [1, 2]}

grid_knn = GridSearchCV(knn_clf, param_grid=parameters, cv=5, refit=True)

# 학습 (grid-search 후)
grid_knn.fit(updrs_x_train_scaled, updrs_y_train)

# GridSearchCV 결과를 추출해 데이터 프레임으로 반환
scores_knn = pd.DataFrame(grid_knn.cv_results_)
scores_knn = scores_knn[['params', 'mean_test_score', 'rank_test_score',
                         'split0_test_score', 'split1_test_score', 'split2_test_score']]

print('KNN GridSearch 최적 파라미터: ', grid_knn.best_params_)
print('KNN Trees GridSearch 최고 점수: ', grid_knn.best_score_)

knn_estimator = grid_knn.best_estimator_

# GridSearchCV의 best_estmator_ 는 이미 최적 학습이 됐으므로 별도 학습이 필요 없음
updrs_pred_knn_after = knn_estimator.predict(updrs_x_test_scaled)
print('grid-search 후 KNN test 데이터세트 정확도: {0: .4f}'.format(accuracy_score(updrs_y_test, updrs_pred_knn_after)))

print('------------------ After Test KNN ------------------------')
print(confusion_matrix(updrs_y_test, updrs_pred_knn_after))
print(classification_report(updrs_y_test, updrs_pred_knn_after, target_names=['class 0', 'class 1']))

# grid search 전 KNN 예측 정확도 실제 y_test / 예측 y_pred:  0.8333333333333334
# ------------------ Before Test KNN ------------------------
# [[116  22]
#  [ 28 134]]
#               precision    recall  f1-score   support
#
#      class 0       0.81      0.84      0.82       138
#      class 1       0.86      0.83      0.84       162
#
#     accuracy                           0.83       300
#    macro avg       0.83      0.83      0.83       300
# weighted avg       0.83      0.83      0.83       300
#
# KNN GridSearch 최적 파라미터:  {'leaf_size': 10, 'n_neighbors': 3, 'p': 1}
# KNN Trees GridSearch 최고 점수:  0.8883333333333333

# grid-search 후 KNN test 데이터세트 정확도:  0.8767
# ------------------ After Test KNN ------------------------
# [[119  19]
#  [ 18 144]]
#               precision    recall  f1-score   support
#
#      class 0       0.87      0.86      0.87       138
#      class 1       0.88      0.89      0.89       162
#
#     accuracy                           0.88       300
#    macro avg       0.88      0.88      0.88       300
# weighted avg       0.88      0.88      0.88       300
















# # 이제 전혀 사용하지 않은 테스트 데이터를 예측
#
# # 테스트 데이터도 스케일링
# updrs_x_test_scaled = rs_scaler.transform(updrs_x_test)
#
# # 베스트 파라미터로 학습된 모델로 테스트 데이터 예측
# updrs_pred_knn_test = knn_estimator.predict(updrs_x_test_scaled)
# print('grid-search 후 KNN 테스트 데이터세트 정확도: {0: .4f}'.format(accuracy_score(updrs_y_test, updrs_pred_knn_test)))
#
# print('------------------ Test KNN ------------------------')
# print(confusion_matrix(updrs_y_test, updrs_pred_knn_test))
# print(classification_report(updrs_y_test, updrs_pred_knn_test, target_names=['class 0', 'class 1', 'class 2']))

#

# # Porter 변환
# porter = Porter(knn_estimator, language='java')
# output = porter.export(embed_data=True)
# # print(output)
#
# # Porter 변환된 결과 파일 저장
# f = open("updrs_KNeighborsClassifier_Robust_3class_210517.java", 'w')
# f.write(output)
# f.close()
#
# # 항상 원본 추정기와 변환 된 추정기 간의 무결성을 확인하고 계산해야 합니다.
# integrity = porter.integrity_score(updrs_x_test_scaled)
# print(integrity)
# # code too large...
