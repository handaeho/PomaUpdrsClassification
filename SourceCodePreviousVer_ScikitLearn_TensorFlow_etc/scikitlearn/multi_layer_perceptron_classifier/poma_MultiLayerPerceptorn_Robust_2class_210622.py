import pickle
import joblib
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler


poma_2class = pd.read_csv('../../../dataset/real_poma_2class_dataset_210518.csv')
# updrs_2class = pd.read_csv('../../dataset/real_updrs_2class_dataset_210518.csv')

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

# MLP 모델 생성
mlp_clf = MLPClassifier(random_state=210518, early_stopping=True)

# train set으로 학습 (grid-search 전)
mlp_clf.fit(poma_x_train_scaled, poma_y_train)

# test set으로 예측 (grid-search 전)
poma_pred_mlp_before = mlp_clf.predict(poma_x_test_scaled)

print('grid search 전 MLP 예측 정확도 실제 y_test / 예측 y_pred: ', accuracy_score(poma_y_test, poma_pred_mlp_before))

print('------------------ Before test MLP ------------------------')
print(confusion_matrix(poma_y_test, poma_pred_mlp_before))
print(classification_report(poma_y_test, poma_pred_mlp_before, target_names=['class 0', 'class 1']))

# 파라미터를 딕셔너리 형태로 설정 --> 이때 파라미터는 모델의 파라미터와 같아야 한다.
parameters = {'alpha': [0.0001], 'momentum': [0.9]}
# parameters = {'solver': ['lbfgs', 'sgd', 'adam'], 'alpha': [0.0001, 0.001, 0.01, 0.1],
#               'learning_rate': ['constant', 'invscaling', 'adaptive'], 'momentum': [0.5, 0.6, 0.7, 0.8, 0.9],
#               'beta_1': [0.7, 0.8, 0.9], 'beta_2': [0.7, 0.8, 0.9]}
# --> 이런 식으로 하면 된다 ~

grid_mlp = GridSearchCV(mlp_clf, param_grid=parameters, cv=5, refit=True)

# 학습 (grid-search 후)
grid_mlp.fit(poma_x_train_scaled, poma_y_train)

# GridSearchCV 결과를 추출해 데이터 프레임으로 반환
scores_mlp = pd.DataFrame(grid_mlp.cv_results_)
scores_mlp = scores_mlp[['params', 'mean_test_score', 'rank_test_score',
                         'split0_test_score', 'split1_test_score', 'split2_test_score']]

print('MLP GridSearch 최적 파라미터: ', grid_mlp.best_params_)
print('MLP GridSearch 최고 점수: ', grid_mlp.best_score_)

mlp_estimator = grid_mlp.best_estimator_

# GridSearchCV의 best_estmator_ 는 이미 최적 학습이 됐으므로 별도 학습이 필요 없음
poma_pred_mlp_after = mlp_estimator.predict(poma_x_test_scaled)
print('grid-search 후 MLP test 데이터세트 정확도: {0: .4f}'.format(accuracy_score(poma_y_test, poma_pred_mlp_after)))

print('------------------ After test MLP ------------------------')
print(confusion_matrix(poma_y_test, poma_pred_mlp_after))
print(classification_report(poma_y_test, poma_pred_mlp_after, target_names=['class 0', 'class 1']))

# ---------------------------------------------------------------------------------------------------------------

# Save Model
saved_model = pickle.dumps(mlp_estimator)

# Load the pickled model
clf_from_pickle = pickle.loads(saved_model)

# Use the loaded pickled model to make predictions (test data)
result_pickle = clf_from_pickle.predict(poma_x_test_scaled)

print('MLP Saved Model POMA Accuracy: {0: .4f}'.format(accuracy_score(poma_y_test, result_pickle)))

print('------------------ Test MLP Pickle Model ------------------------')
print(confusion_matrix(poma_y_test, result_pickle))
print(classification_report(poma_y_test, result_pickle, target_names=['class 0', 'class 1']))

# pickled model로 변환한 모델을 joblib 객체로 저장
joblib.dump(mlp_estimator, 'mlp_poma_2class_210622.pkl')

# ---------------------------------------------------------------------------------------------------------------
