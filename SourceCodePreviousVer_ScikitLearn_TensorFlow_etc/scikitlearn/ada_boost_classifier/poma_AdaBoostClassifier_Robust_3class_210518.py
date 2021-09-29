import pickle

import joblib
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
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
rs_scaler.fit(poma_x_train_t)

# 훈련 데이터 스케일링
poma_x_train_scaled = rs_scaler.transform(poma_x_train_t)

# 검증 데이터의 스케일링
poma_x_eval_scaled = rs_scaler.transform(poma_x_eval)

# 스케일링 된 결과 값으로 본래 값을 구할 수도 있다.
poma_x_origin = rs_scaler.inverse_transform(poma_x_train_scaled)

# AdaBoost 모델의 base estimator ---> Decision Tree
dt_clf = DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=6,
                                splitter='random', random_state=210518)
# -> 먼저 테스트 했던 Decision Tree의 best parameter 사용?

# AdaBoost 모델 생성
ada_clf = AdaBoostClassifier(base_estimator=dt_clf, random_state=210518)

# 학습 (grid-search 전)
ada_clf.fit(poma_x_train_scaled, poma_y_train_t)

# 예측 (grid-search 전)
poma_pred_ada_before = ada_clf.predict(poma_x_eval_scaled)

print('grid search 전 Ada Boost 예측 정확도 실제 y_eval / 예측 y_pred: ', accuracy_score(poma_y_eval, poma_pred_ada_before))

print('------------------Before Eval Ada Boost ------------------------')
print(confusion_matrix(poma_y_eval, poma_pred_ada_before))
print(classification_report(poma_y_eval, poma_pred_ada_before, target_names=['class 0', 'class 1', 'class 2']))

# 파라미터를 딕셔너리 형태로 설정 --> 이때 파라미터는 모델의 파라미터와 같아야 한다.
parameters = {'n_estimators': [150, 200, 300], 'learning_rate': [1.0, 0.1, 0.3]}

grid_ada = GridSearchCV(ada_clf, param_grid=parameters, cv=5, refit=True)

# 학습 (grid-search 후)
grid_ada.fit(poma_x_train_scaled, poma_y_train_t)

# GridSearchCV 결과를 추출해 데이터 프레임으로 반환
scores_ada = pd.DataFrame(grid_ada.cv_results_)
scores_ada = scores_ada[['params', 'mean_test_score', 'rank_test_score',
                         'split0_test_score', 'split1_test_score', 'split2_test_score']]

print('Ada Boost GridSearch 최적 파라미터: ', grid_ada.best_params_)
print('Ada Boost GridSearch 최고 점수: ', grid_ada.best_score_)

ada_estimator = grid_ada.best_estimator_

# GridSearchCV의 best_estmator_ 는 이미 최적 학습이 됐으므로 별도 학습이 필요 없음
poma_pred_ada_after = ada_estimator.predict(poma_x_eval_scaled)

print('grid-search 후 Ada Boost 검증 데이터세트 정확도: {0: .4f}'.format(accuracy_score(poma_y_eval, poma_pred_ada_after)))

print('------------------ After Eval Ada Boost ------------------------')
print(confusion_matrix(poma_y_eval, poma_pred_ada_after))
print(classification_report(poma_y_eval, poma_pred_ada_after, target_names=['class 0', 'class 1', 'class 2']))

# 이제 전혀 사용하지 않은 테스트 데이터를 예측

# 테스트 데이터도 스케일링
poma_x_test_scaled = rs_scaler.transform(poma_x_test)

# 베스트 파라미터로 학습된 모델로 테스트 데이터 예측
poma_pred_ada_test = ada_estimator.predict(poma_x_test_scaled)
print('grid-search 후 Extra Trees 테스트 데이터세트 정확도: {0: .4f}'.format(accuracy_score(poma_y_test, poma_pred_ada_test)))

print('------------------ Test Extra Trees ------------------------')
print(confusion_matrix(poma_y_test, poma_pred_ada_test))
print(classification_report(poma_y_test, poma_pred_ada_test, target_names=['class 0', 'class 1', 'class 2']))


# grid search 전 Ada Boost 예측 정확도 실제 y_eval / 예측 y_pred:  0.9349336702463676
# ------------------Before Eval Ada Boost ------------------------
# [[1514   43   33]
#  [  58 1011    3]
#  [  51   18  435]]
#               precision    recall  f1-score   support
#
#      class 0       0.93      0.95      0.94      1590
#      class 1       0.94      0.94      0.94      1072
#      class 2       0.92      0.86      0.89       504
#
#     accuracy                           0.93      3166
#    macro avg       0.93      0.92      0.93      3166
# weighted avg       0.93      0.93      0.93      3166
#

# Ada Boost GridSearch 최적 파라미터:  {'learning_rate': 1.0, 'n_estimators': 300}
# Ada Boost GridSearch 최고 점수:  0.9447165971576453

# grid-search 후 Ada Boost 검증 데이터세트 정확도:  0.9450
# ------------------ After Eval Ada Boost ------------------------
# [[1536   27   27]
#  [  51 1020    1]
#  [  60    8  436]]
#               precision    recall  f1-score   support
#
#      class 0       0.93      0.97      0.95      1590
#      class 1       0.97      0.95      0.96      1072
#      class 2       0.94      0.87      0.90       504
#
#     accuracy                           0.95      3166
#    macro avg       0.95      0.93      0.94      3166
# weighted avg       0.95      0.95      0.94      3166
#

# grid-search 후 Extra Trees 테스트 데이터세트 정확도:  0.9454
# ------------------ Test Extra Trees ------------------------
# [[1907   47   34]
#  [  50 1286    4]
#  [  66   15  549]]
#               precision    recall  f1-score   support
#
#      class 0       0.94      0.96      0.95      1988
#      class 1       0.95      0.96      0.96      1340
#      class 2       0.94      0.87      0.90       630
#
#     accuracy                           0.95      3958
#    macro avg       0.94      0.93      0.94      3958
# weighted avg       0.95      0.95      0.95      3958

# ---------------------------------------------------------------------------------------------------------------

# Save Model
saved_model = pickle.dumps(ada_estimator)

# Load the pickled model
clf_from_pickle = pickle.loads(saved_model)

# Use the loaded pickled model to make predictions (test data)
result_pickle = clf_from_pickle.predict(poma_x_test_scaled)

print('Ada Boost Saved Model POMA Accuracy: {0: .4f}'.format(accuracy_score(poma_y_test, result_pickle)))

print('------------------ Test Ada Boost Pickle Model ------------------------')
print(confusion_matrix(poma_y_test, result_pickle))
print(classification_report(poma_y_test, result_pickle, target_names=['class 0', 'class 1', 'class 2']))

# pickled model로 변환한 모델을 joblib 객체로 저장
joblib.dump(ada_estimator, 'ada_boost_poma_3class_210615.pkl')

# ---------------------------------------------------------------------------------------------------------------

# # Porter 변환
# porter = Porter(ada_estimator, language='java')
# output = porter.export(embed_data=True)
# # print(output)
#
# # Porter 변환된 결과 파일 저장
# f = open("poma_AdaBoostClassifier_Robust_3class_210518.java", 'w')
# f.write(output)
# f.close()
#
# # 항상 원본 추정기와 변환 된 추정기 간의 무결성을 확인하고 계산해야 합니다.
# integrity = porter.integrity_score(poma_x_test_scaled)
# print(integrity)
# # => error: too many constants
