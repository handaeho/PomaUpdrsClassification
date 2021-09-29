import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier
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

# MLP 모델 생성
mlp_clf = MLPClassifier(random_state=210518, early_stopping=True)

# train set으로 학습 (grid-search 전)
mlp_clf.fit(updrs_x_train_scaled, updrs_y_train)

# test set으로 예측 (grid-search 전)
updrs_pred_mlp_before = mlp_clf.predict(updrs_x_test_scaled)

print('grid search 전 MLP 예측 정확도 실제 y_test / 예측 y_pred: ', accuracy_score(updrs_y_test, updrs_pred_mlp_before))

print('------------------ Before test MLP ------------------------')
print(confusion_matrix(updrs_y_test, updrs_pred_mlp_before))
print(classification_report(updrs_y_test, updrs_pred_mlp_before, target_names=['class 0', 'class 1']))

# 파라미터를 딕셔너리 형태로 설정 --> 이때 파라미터는 모델의 파라미터와 같아야 한다.
parameters = {'solver': ['lbfgs', 'sgd', 'adam'], 'alpha': [0.0001, 0.001, 0.01, 0.1],
              'momentum': [0.3, 0.4, 0.5, 0.6, 0.7]}
# parameters = {'solver': ['lbfgs', 'sgd', 'adam'], 'alpha': [0.0001, 0.001, 0.01, 0.1],
#               'learning_rate': ['constant', 'invscaling', 'adaptive'], 'momentum': [0.5, 0.6, 0.7, 0.8, 0.9],
#               'beta_1': [0.7, 0.8, 0.9], 'beta_2': [0.7, 0.8, 0.9]}
# --> 이런 식으로 하면 된다 ~

grid_mlp = GridSearchCV(mlp_clf, param_grid=parameters, cv=5, refit=True)

# 학습 (grid-search 후)
grid_mlp.fit(updrs_x_train_scaled, updrs_y_train)

# GridSearchCV 결과를 추출해 데이터 프레임으로 반환
scores_mlp = pd.DataFrame(grid_mlp.cv_results_)
scores_mlp = scores_mlp[['params', 'mean_test_score', 'rank_test_score',
                         'split0_test_score', 'split1_test_score', 'split2_test_score']]

print('MLP GridSearch 최적 파라미터: ', grid_mlp.best_params_)
print('MLP GridSearch 최고 점수: ', grid_mlp.best_score_)

mlp_estimator = grid_mlp.best_estimator_

# GridSearchCV의 best_estmator_ 는 이미 최적 학습이 됐으므로 별도 학습이 필요 없음
updrs_pred_mlp_after = mlp_estimator.predict(updrs_x_test_scaled)

print('grid-search 후 MLP test 데이터세트 정확도: {0: .4f}'.format(accuracy_score(updrs_y_test, updrs_pred_mlp_after)))

print('------------------ After test MLP ------------------------')
print(confusion_matrix(updrs_y_test, updrs_pred_mlp_after))
print(classification_report(updrs_y_test, updrs_pred_mlp_after, target_names=['class 0', 'class 1']))

# grid search 전 MLP 예측 정확도 실제 y_test / 예측 y_pred:  0.78
# ------------------ Before test MLP ------------------------
# [[102  36]
#  [ 30 132]]
#               precision    recall  f1-score   support
#
#      class 0       0.77      0.74      0.76       138
#      class 1       0.79      0.81      0.80       162
#
#     accuracy                           0.78       300
#    macro avg       0.78      0.78      0.78       300
# weighted avg       0.78      0.78      0.78       300

# MLP GridSearch 최적 파라미터:  {'alpha': 0.1, 'momentum': 0.3, 'solver': 'lbfgs'}
# MLP GridSearch 최고 점수:  0.8591666666666666
# grid-search 후 MLP test 데이터세트 정확도:  0.8567
# ------------------ After test MLP ------------------------
# [[111  27]
#  [ 16 146]]
#               precision    recall  f1-score   support
#
#      class 0       0.87      0.80      0.84       138
#      class 1       0.84      0.90      0.87       162
#
#     accuracy                           0.86       300
#    macro avg       0.86      0.85      0.85       300
# weighted avg       0.86      0.86      0.86       300
