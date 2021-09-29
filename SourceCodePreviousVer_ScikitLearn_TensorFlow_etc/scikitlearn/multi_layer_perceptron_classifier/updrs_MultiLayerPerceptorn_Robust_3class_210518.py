import pickle

import joblib
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier
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

# MLP 모델 생성
mlp_clf = MLPClassifier(random_state=210518, early_stopping=True)

# train set으로 학습 (grid-search 전)
mlp_clf.fit(updrs_x_train_scaled, updrs_y_train_t)

# eval set으로 예측 검증(평가) (grid-search 전)
updrs_pred_mlp_before = mlp_clf.predict(updrs_x_eval_scaled)

print('grid search 전 MLP 예측 정확도 실제 y_eval / 예측 y_pred: ', accuracy_score(updrs_y_eval, updrs_pred_mlp_before))

print('------------------ Before Eval MLP ------------------------')
print(confusion_matrix(updrs_y_eval, updrs_pred_mlp_before))
print(classification_report(updrs_y_eval, updrs_pred_mlp_before, target_names=['class 0', 'class 1', 'class 2']))

# 파라미터를 딕셔너리 형태로 설정 --> 이때 파라미터는 모델의 파라미터와 같아야 한다.
parameters = {'alpha': [0.0001, 0.001, 0.01, 0.1], 'momentum': [0.3, 0.4, 0.5, 0.6, 0.7]}
# parameters = {'solver': ['lbfgs', 'sgd', 'adam'], 'alpha': [0.0001, 0.001, 0.01, 0.1],
#               'learning_rate': ['constant', 'invscaling', 'adaptive'], 'momentum': [0.5, 0.6, 0.7, 0.8, 0.9],
#               'beta_1': [0.7, 0.8, 0.9], 'beta_2': [0.7, 0.8, 0.9]}
# --> 이런 식으로 하면 된다 ~


grid_mlp = GridSearchCV(mlp_clf, param_grid=parameters, cv=5, refit=True)

# 학습 (grid-search 후)
grid_mlp.fit(updrs_x_train_scaled, updrs_y_train_t)

# GridSearchCV 결과를 추출해 데이터 프레임으로 반환
scores_mlp = pd.DataFrame(grid_mlp.cv_results_)
scores_mlp = scores_mlp[['params', 'mean_test_score', 'rank_test_score',
                         'split0_test_score', 'split1_test_score', 'split2_test_score']]

print('MLP GridSearch 최적 파라미터: ', grid_mlp.best_params_)
print('MLP GridSearch 최고 점수: ', grid_mlp.best_score_)

mlp_estimator = grid_mlp.best_estimator_

# GridSearchCV의 best_estmator_ 는 이미 최적 학습이 됐으므로 별도 학습이 필요 없음
updrs_pred_mlp_after = mlp_estimator.predict(updrs_x_eval_scaled)
print('grid-search 후 MLP 검증 데이터세트 정확도: {0: .4f}'.format(accuracy_score(updrs_y_eval, updrs_pred_mlp_after)))

print('------------------ After Eval MLP ------------------------')
print(confusion_matrix(updrs_y_eval, updrs_pred_mlp_after))
print(classification_report(updrs_y_eval, updrs_pred_mlp_after, target_names=['class 0', 'class 1', 'class 2']))

# 이제 전혀 사용하지 않은 테스트 데이터를 예측

# 테스트 데이터도 스케일링
updrs_x_test_scaled = rs_scaler.transform(updrs_x_test)

# 베스트 파라미터로 학습된 모델로 테스트 데이터 예측
updrs_pred_mlp_test = mlp_estimator.predict(updrs_x_test_scaled)
print('grid-search 후 MLP 테스트 데이터세트 정확도: {0: .4f}'.format(accuracy_score(updrs_y_test, updrs_pred_mlp_test)))

print('------------------ Test MLP ------------------------')
print(confusion_matrix(updrs_y_test, updrs_pred_mlp_test))
print(classification_report(updrs_y_test, updrs_pred_mlp_test, target_names=['class 0', 'class 1', 'class 2']))


# grid search 전 MLP 예측 정확도 실제 y_eval / 예측 y_pred:  0.9150347441566645
# ------------------ Before Eval MLP ------------------------
# [[1363   75   12]
#  [  94 1093   35]
#  [  23   30  441]]
#               precision    recall  f1-score   support
#
#      class 0       0.92      0.94      0.93      1450
#      class 1       0.91      0.89      0.90      1222
#      class 2       0.90      0.89      0.90       494
#
#     accuracy                           0.92      3166
#    macro avg       0.91      0.91      0.91      3166
# weighted avg       0.91      0.92      0.91      3166

# MLP GridSearch 최적 파라미터:  {'alpha': 0.001, 'momentum': 0.3}
# MLP GridSearch 최고 점수:  0.9156536560996738

# grid-search 후 MLP 검증 데이터세트 정확도:  0.9150
# ------------------ After Eval MLP ------------------------
# [[1364   76   10]
#  [  93 1093   36]
#  [  22   32  440]]
#               precision    recall  f1-score   support
#
#      class 0       0.92      0.94      0.93      1450
#      class 1       0.91      0.89      0.90      1222
#      class 2       0.91      0.89      0.90       494
#
#     accuracy                           0.92      3166
#    macro avg       0.91      0.91      0.91      3166
# weighted avg       0.91      0.92      0.91      3166
#

# grid-search 후 MLP 테스트 데이터세트 정확도:  0.9255
# ------------------ Test MLP ------------------------
# [[1706   91   15]
#  [  98 1400   30]
#  [  30   31  557]]
#               precision    recall  f1-score   support
#
#      class 0       0.93      0.94      0.94      1812
#      class 1       0.92      0.92      0.92      1528
#      class 2       0.93      0.90      0.91       618
#
#     accuracy                           0.93      3958
#    macro avg       0.93      0.92      0.92      3958
# weighted avg       0.93      0.93      0.93      3958

# ---------------------------------------------------------------------------------------------------------------

# Save Model
saved_model = pickle.dumps(mlp_estimator)

# Load the pickled model
clf_from_pickle = pickle.loads(saved_model)

# Use the loaded pickled model to make predictions (test data)
result_pickle = clf_from_pickle.predict(updrs_x_test_scaled)

print('MLP Saved Model UPDRS Accuracy: {0: .4f}'.format(accuracy_score(updrs_y_test, result_pickle)))

print('------------------ Test MLP Pickle Model ------------------------')
print(confusion_matrix(updrs_y_test, result_pickle))
print(classification_report(updrs_y_test, result_pickle, target_names=['class 0', 'class 1', 'class 2']))

# pickled model로 변환한 모델을 joblib 객체로 저장
joblib.dump(mlp_estimator, 'mlp_updrs_3class_210615.pkl')

# ---------------------------------------------------------------------------------------------------------------

# # Porter 변환
# porter = Porter(mlp_estimator, language='java')
# output = porter.export(embed_data=True)
# # print(output)
#
# # Porter 변환된 결과 파일 저장
# f = open("updrs_MultiLayerPerceptorn_Robust_3class_210518.java", 'w')
# f.write(output)
# f.close()
#
# # 항상 원본 추정기와 변환 된 추정기 간의 무결성을 확인하고 계산해야 합니다.
# integrity = porter.integrity_score(updrs_x_test_scaled)
# print(integrity)
# # code too large...
