import pickle
import joblib
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler


# poma_2class = pd.read_csv('../../dataset/real_poma_2class_dataset_210518.csv')
updrs_2class = pd.read_csv('../../../dataset/real_updrs_2class_dataset_210518.csv')

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

# Extra Trees 모델 생성
et_clf = ExtraTreesClassifier(random_state=210518)

# 학습 (grid-search 전)
et_clf.fit(updrs_x_train_scaled, updrs_y_train)

# 예측 (grid-search 전)
updrs_pred_et_before = et_clf.predict(updrs_x_test_scaled)

print('grid search 전 Extra Trees 예측 정확도 실제 y_test / 예측 y_pred: ', accuracy_score(updrs_y_test, updrs_pred_et_before))

print('------------------ Before Test Extra Trees ------------------------')
print(confusion_matrix(updrs_y_test, updrs_pred_et_before))
print(classification_report(updrs_y_test, updrs_pred_et_before, target_names=['class 0', 'class 1']))

# 파라미터를 딕셔너리 형태로 설정 --> 이때 파라미터는 모델의 파라미터와 같아야 한다.
# parameters = {'n_estimators': [100, 200, 300, 400, 500], 'criterion': ['gini', 'entropy'],
#               'max_depth': [None, 2, 3, 4, 5], 'min_samples_split': [2, 3, 4, 5, 6],
#               'min_samples_leaf': [1, 2, 3.1, 4.1, 5.1], 'min_weight_fraction_leaf': [0.0],
#               'max_features': ['auto', 'sqrt', 'log2', None, 1, 1.5], 'max_leaf_nodes': [None, 1, 2, 3, 4],
#               'min_impurity_decrease': [0.0, 1.0, 2.0, 3.0, 4.0], 'bootstrap': [False, True],
#               'oob_score': [False, True], 'n_jobs': [None, 1, 2, 3, 4],
#               'warm_start': [False, True], 'class_weight': [None, 'balanced', 'balanced_subsample”'],
#               'ccp_alpha': [0.0, 0.2, 0.4, 0.6, 0.8], 'max_samples': [None, 0.1, 0.5, 0.9]}
# --> 이런식으로 쓰면 된다 ~
parameters = {'n_estimators': [100, 200, 300], 'min_samples_split': [2, 3, 4, 5, 6],
              'min_impurity_decrease': [0.0, 1.0, 2.0, 3.0], 'max_samples': [None, 0.1, 0.5, 0.9]}

grid_et = GridSearchCV(et_clf, param_grid=parameters, cv=5, refit=True)

# 학습 (grid-search 후)
grid_et.fit(updrs_x_train_scaled, updrs_y_train)

# GridSearchCV 결과를 추출해 데이터 프레임으로 반환
scores_et = pd.DataFrame(grid_et.cv_results_)
scores_et = scores_et[['params', 'mean_test_score', 'rank_test_score',
                       'split0_test_score', 'split1_test_score', 'split2_test_score']]

print('Extra Trees GridSearch 최적 파라미터: ', grid_et.best_params_)
print('Extra Trees GridSearch 최고 점수: ', grid_et.best_score_)

et_estimator = grid_et.best_estimator_

# GridSearchCV의 best_estmator_ 는 이미 최적 학습이 됐으므로 별도 학습이 필요 없음
updrs_pred_et_after = et_estimator.predict(updrs_x_test_scaled)
print('grid-search 후 Extra Trees test 데이터세트 정확도: {0: .4f}'.format(accuracy_score(updrs_y_test, updrs_pred_et_after)))

print('------------------ After Test Extra Trees ------------------------')
print(confusion_matrix(updrs_y_test, updrs_pred_et_after))
print(classification_report(updrs_y_test, updrs_pred_et_after, target_names=['class 0', 'class 1']))

# ---------------------------------------------------------------------------------------------------------------

# Save Model
saved_model = pickle.dumps(et_estimator)

# Load the pickled model
clf_from_pickle = pickle.loads(saved_model)

# Use the loaded pickled model to make predictions (test data)
result_pickle = clf_from_pickle.predict(updrs_x_test_scaled)

print('ET Saved Model POMA Accuracy: {0: .4f}'.format(accuracy_score(updrs_y_test, result_pickle)))

print('------------------ Test ET Pickle Model ------------------------')
print(confusion_matrix(updrs_y_test, result_pickle))
print(classification_report(updrs_y_test, result_pickle, target_names=['class 0', 'class 1']))

# pickled model로 변환한 모델을 joblib 객체로 저장
joblib.dump(et_estimator, 'et_updrs_2class_210622.pkl')

# ---------------------------------------------------------------------------------------------------------------
