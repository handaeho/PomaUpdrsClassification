import pandas as pd
import pickle

from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import NuSVC
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

# linear, nu SVC 모델 생성
nu_svc_clf = NuSVC(nu=0.3, kernel='rbf', gamma='auto', random_state=210518)

# 학습 (grid-search 전)
nu_svc_clf.fit(updrs_x_train_scaled, updrs_y_train_t)

# 예측 (grid-search 전)
updrs_pred_nu_before = nu_svc_clf.predict(updrs_x_eval_scaled)

print('grid search 전 Nu 예측 정확도 실제 y_eval / 예측 pred: ', accuracy_score(updrs_y_eval, updrs_pred_nu_before))

print('----------------- Before NU ---------------------')
print(confusion_matrix(updrs_y_eval, updrs_pred_nu_before))
print(classification_report(updrs_y_eval, updrs_pred_nu_before, target_names=['class 0', 'class 1', 'class 2']))

# 파라미터를 딕셔너리 형태로 설정 --> 이때 파라미터는 모델의 파라미터와 같아야 한다.
# nu_parameters = {'nu': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'gamma': ['scale', 'auto']}
nu_parameters = {'nu': [0.3, 0.4, 0.5, 0.6, 0.7]}

# grid-search
grid_nu_svc = GridSearchCV(nu_svc_clf, param_grid=nu_parameters, cv=5, refit=True)

# 학습 (grid-search 후)
grid_nu_svc.fit(updrs_x_train_scaled, updrs_y_train_t)

# GridSearchCV 결과를 추출해 데이터 프레임으로 반환
scores_df_nu = pd.DataFrame(grid_nu_svc.cv_results_)
scores_df_nu = scores_df_nu[['params', 'mean_test_score', 'rank_test_score',
                             'split0_test_score', 'split1_test_score', 'split2_test_score']]

# print(scores_df_nu)

print('Nu GridSearch 최적 파라미터: ', grid_nu_svc.best_params_)
print('Nu GridSearch 최고 점수: ', grid_nu_svc.best_score_)

nu_estimator = grid_nu_svc.best_estimator_

# GridSearchCV의 best_estmator_ 는 이미 최적 학습이 됐으므로 별도 학습이 필요 없음
updrs_pred_nu_after = nu_estimator.predict(updrs_x_eval_scaled)
print('grid-search 후 Nu 검증 데이터세트 정확도: {0: .4f}'.format(accuracy_score(updrs_y_eval, updrs_pred_nu_after)))

print('----------------- After NU ---------------------')
print(confusion_matrix(updrs_y_eval, updrs_pred_nu_after))
print(classification_report(updrs_y_eval, updrs_pred_nu_after, target_names=['class 0', 'class 1', 'class 2']))

# 이제 전혀 사용하지 않은 테스트 데이터를 예측

# 테스트 데이터도 스케일링
updrs_x_test_scaled = rs_scaler.transform(updrs_x_test)

# 베스트 파라미터로 학습된 모델로 테스트 데이터 예측
updrs_pred_nu_test = nu_estimator.predict(updrs_x_test_scaled)
print('grid-search 후 Nu 테스트 데이터세트 정확도: {0: .4f}'.format(accuracy_score(updrs_y_test, updrs_pred_nu_test)))

print('----------------- Test NU ---------------------')
print(confusion_matrix(updrs_y_test, updrs_pred_nu_test))
print(classification_report(updrs_y_test, updrs_pred_nu_test, target_names=['class 0', 'class 1', 'class 2']))

# ---------------------------------------------------------------------------------------------------------------

# grid search 전 Nu 예측 정확도 실제 y_eval / 예측 pred:  0.8900821225521163
# ----------------- Before NU ---------------------
# [[1360   82    8]
#  [  99 1088   35]
#  [  96   28  370]]
#               precision    recall  f1-score   support
#
#      class 0       0.87      0.94      0.91      1450
#      class 1       0.91      0.89      0.90      1222
#      class 2       0.90      0.75      0.82       494
#
#     accuracy                           0.89      3166
#    macro avg       0.89      0.86      0.87      3166
# weighted avg       0.89      0.89      0.89      3166

# Nu GridSearch 최적 파라미터:  {'nu': 0.3}
# Nu GridSearch 최고 점수:  0.8917232499412183

# grid-search 후 Nu 검증 데이터세트 정확도:  0.8901
# ----------------- After NU ---------------------
# [[1360   82    8]
#  [  99 1088   35]
#  [  96   28  370]]
#               precision    recall  f1-score   support
#
#      class 0       0.87      0.94      0.91      1450
#      class 1       0.91      0.89      0.90      1222
#      class 2       0.90      0.75      0.82       494
#
#     accuracy                           0.89      3166
#    macro avg       0.89      0.86      0.87      3166
# weighted avg       0.89      0.89      0.89      3166
#
# grid-search 후 Nu 테스트 데이터세트 정확도:  0.9037
# ----------------- Test NU ---------------------
# [[1702  102    8]
#  [  93 1396   39]
#  [ 102   37  479]]
#               precision    recall  f1-score   support
#
#      class 0       0.90      0.94      0.92      1812
#      class 1       0.91      0.91      0.91      1528
#      class 2       0.91      0.78      0.84       618
#
#     accuracy                           0.90      3958
#    macro avg       0.91      0.88      0.89      3958
# weighted avg       0.90      0.90      0.90      3958

# ---------------------------------------------------------------------------------------------------------------

# Save Model
saved_model = pickle.dumps(nu_estimator)

# Load the pickled model
clf_from_pickle = pickle.loads(saved_model)

# Use the loaded pickled model to make predictions (test data)
result_pickle = clf_from_pickle.predict(updrs_x_test_scaled)

print('NuSVC Saved Model UPDRS Accuracy: {0: .4f}'.format(accuracy_score(updrs_y_test, result_pickle)))

print('------------------ Test Nu SVC Pickle Model ------------------------')
print(confusion_matrix(updrs_y_test, result_pickle))
print(classification_report(updrs_y_test, result_pickle, target_names=['class 0', 'class 1', 'class 2']))

# pickled model로 변환한 모델을 joblib 객체로 저장
joblib.dump(nu_estimator, 'nu_svc_updrs_3class_210615.pkl')

# ---------------------------------------------------------------------------------------------------------------

# # Porter 변환
# porter_nu = Porter(nu_estimator, language='java')
#
# output_nu = porter_nu.export(embed_data=True)
#
# # Porter 변환된 결과 파일 저장
# f_nu = open("updrs_RobustScalerSVC_3class_nu_210518.java", 'w')
#
# f_nu.write(output_nu)
#
# f_nu.close()
#
# # 항상 원본 추정기와 변환 된 추정기 간의 무결성을 확인하고 계산해야 합니다.
# integrity_nu = porter_nu.integrity_score(updrs_x_test_scaled)
# print(integrity_nu)



