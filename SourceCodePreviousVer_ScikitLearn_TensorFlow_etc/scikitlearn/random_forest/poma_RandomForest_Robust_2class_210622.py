import pickle
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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

# Decision Tree 모델 생성
rf_clf = RandomForestClassifier(random_state=210518)

# 학습 (grid-search 전)
rf_clf.fit(poma_x_train_scaled, poma_y_train)

# 예측 (grid-search 전)
poma_pred_rf_before = rf_clf.predict(poma_x_test_scaled)

print('grid search 전 RF 예측 정확도 실제 poma_y_test / 예측 poma_y_pred: ', accuracy_score(poma_y_test, poma_pred_rf_before))

print('------------------ Before test d-Tree ------------------------')
print(confusion_matrix(poma_y_test, poma_pred_rf_before))
print(classification_report(poma_y_test, poma_pred_rf_before, target_names=['class 0', 'class 1']))

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
grid_rf.fit(poma_x_train_scaled, poma_y_train)

# GridSearchCV 결과를 추출해 데이터 프레임으로 반환
scores_rf = pd.DataFrame(grid_rf.cv_results_)
scores_tf = scores_rf[['params', 'mean_test_score', 'rank_test_score',
                       'split0_test_score', 'split1_test_score', 'split2_test_score']]

# print(scores_dtree)

print('Random Forest GridSearch 최적 파라미터: ', grid_rf.best_params_)
print('Random Forest GridSearch 최고 점수: ', grid_rf.best_score_)

rf_estimator = grid_rf.best_estimator_

# GridSearchCV의 best_estmator_ 는 이미 최적 학습이 됐으므로 별도 학습이 필요 없음
pred_rf_after = rf_estimator.predict(poma_x_test_scaled)
print('grid-search 후 RF test 데이터세트 정확도: {0: .4f}'.format(accuracy_score(poma_y_test, pred_rf_after)))

print('------------------ After test d-Tree ------------------------')
print(confusion_matrix(poma_y_test, pred_rf_after))
print(classification_report(poma_y_test, pred_rf_after, target_names=['class 0', 'class 1']))

# ---------------------------------------------------------------------------------------------------------------

# Save Model
saved_model = pickle.dumps(rf_estimator)

# Load the pickled model
clf_from_pickle = pickle.loads(saved_model)

# Use the loaded pickled model to make predictions (test data)
result_pickle = clf_from_pickle.predict(poma_x_test_scaled)

print('RF Saved Model POMA Accuracy: {0: .4f}'.format(accuracy_score(poma_y_test, result_pickle)))

print('------------------ Test RF Pickle Model ------------------------')
print(confusion_matrix(poma_y_test, result_pickle))
print(classification_report(poma_y_test, result_pickle, target_names=['class 0', 'class 1']))

# pickled model로 변환한 모델을 joblib 객체로 저장
joblib.dump(rf_estimator, 'rf_poma_2class_210622.pkl')

# ---------------------------------------------------------------------------------------------------------------

