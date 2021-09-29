import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler


poma_2class = pd.read_csv('../../../dataset/poma_dataset_210518_1500.csv')
# updrs_2class = pd.read_csv('../../dataset/updrs_dataset_210518_1500.csv')

poma_dataset_2class = poma_2class.copy()
# updrs_dataset_2class = updrs_2class.copy()

print(poma_dataset_2class)
# print(updrs_dataset_2class)

poma_features = poma_dataset_2class
poma_labels = poma_dataset_2class.pop('poma_danger_2class')

# updrs_features = updrs_dataset_2class
# updrs_labels = updrs_dataset_2class.pop('updrs_danger_2class')

poma_x_train, poma_x_test, poma_y_train, poma_y_test = train_test_split(poma_features, poma_labels,
                                                                        test_size=0.2,
                                                                        stratify=poma_labels,
                                                                        shuffle=True,
                                                                        random_state=1234)

print(poma_x_train.shape, poma_y_train.shape)
print(poma_x_test.shape, poma_y_test.shape)

# 변형 객체 생성
rs_scaler = RobustScaler()

# 훈련데이터의 모수 분포 저장
rs_scaler.fit(poma_x_train)

# 훈련 데이터 스케일링
poma_x_train_scaled = rs_scaler.transform(poma_x_train)

# test 데이터의 스케일링
poma_x_test_scaled = rs_scaler.transform(poma_x_test)

# 스케일링 된 결과 값으로 본래 값을 구할 수도 있다.
poma_x_origin = rs_scaler.inverse_transform(poma_x_train_scaled)

# AdaBoost 모델 생성
knn_clf = KNeighborsClassifier()

# 학습 (grid-search 전)
knn_clf.fit(poma_x_train_scaled, poma_y_train)

# 예측 (grid-search 전)
poma_pred_knn_before = knn_clf.predict(poma_x_test_scaled)

print('grid search 전 KNN 예측 정확도 실제 y_test / 예측 y_pred: ', accuracy_score(poma_y_test, poma_pred_knn_before))

print('------------------ Before Test KNN ------------------------')
print(confusion_matrix(poma_y_test, poma_pred_knn_before))
print(classification_report(poma_y_test, poma_pred_knn_before, target_names=['class 0', 'class 1']))

# 파라미터를 딕셔너리 형태로 설정 --> 이때 파라미터는 모델의 파라미터와 같아야 한다.
parameters = {'n_neighbors': [3, 5, 7, 9, 11, 13], 'leaf_size': [10, 30, 50, 70, 90, 110], 'p': [1, 2]}

grid_knn = GridSearchCV(knn_clf, param_grid=parameters, cv=5, refit=True)

# 학습 (grid-search 후)
grid_knn.fit(poma_x_train_scaled, poma_y_train)

# GridSearchCV 결과를 추출해 데이터 프레임으로 반환
scores_knn = pd.DataFrame(grid_knn.cv_results_)
scores_knn = scores_knn[['params', 'mean_test_score', 'rank_test_score',
                         'split0_test_score', 'split1_test_score', 'split2_test_score']]

print('KNN GridSearch 최적 파라미터: ', grid_knn.best_params_)
print('KNN Trees GridSearch 최고 점수: ', grid_knn.best_score_)

knn_estimator = grid_knn.best_estimator_

# GridSearchCV의 best_estmator_ 는 이미 최적 학습이 됐으므로 별도 학습이 필요 없음
poma_pred_knn_after = knn_estimator.predict(poma_x_test_scaled)
print('grid-search 후 KNN test 데이터세트 정확도: {0: .4f}'.format(accuracy_score(poma_y_test, poma_pred_knn_after)))

print('------------------ After Test KNN ------------------------')
print(confusion_matrix(poma_y_test, poma_pred_knn_after))
print(classification_report(poma_y_test, poma_pred_knn_after, target_names=['class 0', 'class 1']))

# grid search 전 KNN 예측 정확도 실제 y_test / 예측 y_pred:  0.8866666666666667
# ------------------ Before Test KNN ------------------------
# [[130  17]
#  [ 17 136]]
#               precision    recall  f1-score   support
#
#      class 0       0.88      0.88      0.88       147
#      class 1       0.89      0.89      0.89       153
#
#     accuracy                           0.89       300
#    macro avg       0.89      0.89      0.89       300
# weighted avg       0.89      0.89      0.89       300
#
# KNN GridSearch 최적 파라미터:  {'leaf_size': 10, 'n_neighbors': 3, 'p': 1}
# KNN Trees GridSearch 최고 점수:  0.885
# grid-search 후 KNN test 데이터세트 정확도:  0.9100
# ------------------ After Test KNN ------------------------
# [[133  14]
#  [ 13 140]]
#               precision    recall  f1-score   support
#
#      class 0       0.91      0.90      0.91       147
#      class 1       0.91      0.92      0.91       153
#
#     accuracy                           0.91       300
#    macro avg       0.91      0.91      0.91       300
# weighted avg       0.91      0.91      0.91       300
