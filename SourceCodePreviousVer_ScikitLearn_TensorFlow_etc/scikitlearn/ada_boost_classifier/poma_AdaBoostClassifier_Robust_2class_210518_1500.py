import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
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

print(poma_x_train.shape, poma_y_train.shape)       # (1200, 95) (1200,)
print(poma_x_test.shape, poma_y_test.shape)         # (300, 95) (300,)

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

# AdaBoost 모델의 base estimator ---> Decision Tree
dt_clf = DecisionTreeClassifier(max_depth=1, max_features=0.3, class_weight='balanced', random_state=210518)

# AdaBoost 모델 생성
ada_clf = AdaBoostClassifier(base_estimator=dt_clf, random_state=210518)

# 학습 (grid-search 전)
ada_clf.fit(poma_x_train_scaled, poma_y_train)

# 예측 (grid-search 전)
poma_pred_ada = ada_clf.predict(poma_x_test_scaled)

print('grid search 전 Ada Boost 예측 정확도 실제 y_eval / 예측 y_pred: ', accuracy_score(poma_y_test, poma_pred_ada))

print('------------------ Before Test Ada Boost ------------------------')
print(confusion_matrix(poma_y_test, poma_pred_ada))
print(classification_report(poma_y_test, poma_pred_ada, target_names=['class 0', 'class 1']))

# 파라미터를 딕셔너리 형태로 설정 --> 이때 파라미터는 모델의 파라미터와 같아야 한다.
parameters = {'n_estimators': [50, 100, 150], 'learning_rate': [1.0, 0.1, 0.5]}

grid_ada = GridSearchCV(ada_clf, param_grid=parameters, cv=5, refit=True)

# 학습 (grid-search 후)
grid_ada.fit(poma_x_train_scaled, poma_y_train)

# GridSearchCV 결과를 추출해 데이터 프레임으로 반환
scores_ada = pd.DataFrame(grid_ada.cv_results_)
scores_ada = scores_ada[['params', 'mean_test_score', 'rank_test_score',
                         'split0_test_score', 'split1_test_score', 'split2_test_score']]

print('Extra Trees GridSearch 최적 파라미터: ', grid_ada.best_params_)
print('Extra Trees GridSearch 최고 점수: ', grid_ada.best_score_)

ada_estimator = grid_ada.best_estimator_

# GridSearchCV의 best_estmator_ 는 이미 최적 학습이 됐으므로 별도 학습이 필요 없음
poma_pred_ada_test = ada_estimator.predict(poma_x_test_scaled)
print('grid-search 후 Ada Boost 검증 데이터세트 정확도: {0: .4f}'.format(accuracy_score(poma_y_test, poma_pred_ada_test)))

print('------------------ After Test Ada Boost ------------------------')
print(confusion_matrix(poma_y_test, poma_pred_ada_test))
print(classification_report(poma_y_test, poma_pred_ada_test, target_names=['class 0', 'class 1']))

# grid search 전 Ada Boost 예측 정확도 실제 y_eval / 예측 y_pred:  0.7766666666666666
# ------------------ Before Test Ada Boost ------------------------
# [[118  29]
#  [ 38 115]]
#               precision    recall  f1-score   support
#
#      class 0       0.76      0.80      0.78       147
#      class 1       0.80      0.75      0.77       153
#
#     accuracy                           0.78       300
#    macro avg       0.78      0.78      0.78       300
# weighted avg       0.78      0.78      0.78       300
#

# Extra Trees GridSearch 최적 파라미터:  {'learning_rate': 1.0, 'n_estimators': 100}
# Extra Trees GridSearch 최고 점수:  0.7941666666666667

# grid-search 후 Ada Boost 검증 데이터세트 정확도:  0.8033
# ------------------ After Test Ada Boost ------------------------
# [[120  27]
#  [ 32 121]]
#               precision    recall  f1-score   support
#
#      class 0       0.79      0.82      0.80       147
#      class 1       0.82      0.79      0.80       153
#
#     accuracy                           0.80       300
#    macro avg       0.80      0.80      0.80       300
# weighted avg       0.80      0.80      0.80       300
