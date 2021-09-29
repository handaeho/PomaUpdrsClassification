import pickle
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
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

# Decision Tree 모델 생성
dtree_clf = DecisionTreeClassifier(random_state=210518)

# 학습 (grid-search 전)
dtree_clf.fit(updrs_x_train_scaled, updrs_y_train)

# 예측 (grid-search 전)
updrs_pred_dtree = dtree_clf.predict(updrs_x_test_scaled)

print('grid search 전 DT 예측 정확도 실제 y_test / 예측 y_pred: ', accuracy_score(updrs_y_test, updrs_pred_dtree))

# 파라미터를 딕셔너리 형태로 설정 --> 이때 파라미터는 모델의 파라미터와 같아야 한다.
parameters = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'],
              'max_depth': [None, 1, 2, 3, 4, 5], 'min_samples_split': [2, 3, 4, 5, 6]}

grid_dtree = GridSearchCV(dtree_clf, param_grid=parameters, cv=5, refit=True)

# 학습 (grid-search 후)
grid_dtree.fit(updrs_x_train_scaled, updrs_y_train)

# GridSearchCV 결과를 추출해 데이터 프레임으로 반환
scores_dtree = pd.DataFrame(grid_dtree.cv_results_)
scores_dtree = scores_dtree[['params', 'mean_test_score', 'rank_test_score',
                             'split0_test_score', 'split1_test_score', 'split2_test_score']]

# print(scores_dtree)

print('Decision Tree GridSearch 최적 파라미터: ', grid_dtree.best_params_)
print('Decision Tree GridSearch 최고 점수: ', grid_dtree.best_score_)

dtree_estimator = grid_dtree.best_estimator_

# GridSearchCV의 best_estmator_ 는 이미 최적 학습이 됐으므로 별도 학습이 필요 없음
updrs_pred_dtree_test = dtree_estimator.predict(updrs_x_test_scaled)
print('grid-search 후 DT test 데이터세트 정확도: {0: .4f}'.format(accuracy_score(updrs_y_test, updrs_pred_dtree_test)))

print('------------------ Test d-Tree ------------------------')
print(confusion_matrix(updrs_y_test, updrs_pred_dtree_test))
print(classification_report(updrs_y_test, updrs_pred_dtree_test, target_names=['class 0', 'class 1']))

# ---------------------------------------------------------------------------------------------------------------

# Save Model
saved_model = pickle.dumps(dtree_estimator)

# Load the pickled model
clf_from_pickle = pickle.loads(saved_model)

# Use the loaded pickled model to make predictions (test data)
result_pickle = clf_from_pickle.predict(updrs_x_test_scaled)

print('DT Saved Model POMA Accuracy: {0: .4f}'.format(accuracy_score(updrs_y_test, result_pickle)))

print('------------------ Test DT Pickle Model ------------------------')
print(confusion_matrix(updrs_y_test, result_pickle))
print(classification_report(updrs_y_test, result_pickle, target_names=['class 0', 'class 1']))

# pickled model로 변환한 모델을 joblib 객체로 저장
joblib.dump(dtree_estimator, 'dt_updrs_2class_210622.pkl')

# ---------------------------------------------------------------------------------------------------------------

