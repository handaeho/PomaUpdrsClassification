import pickle
import joblib
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
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

# ?????? ?????? ??????
rs_scaler = RobustScaler()

# ?????????????????? ?????? ?????? ??????
rs_scaler.fit(updrs_x_train)

# ?????? ????????? ????????????
updrs_x_train_scaled = rs_scaler.transform(updrs_x_train)

# test ???????????? ????????????
updrs_x_test_scaled = rs_scaler.transform(updrs_x_test)

# ???????????? ??? ?????? ????????? ?????? ?????? ?????? ?????? ??????.
updrs_x_origin = rs_scaler.inverse_transform(updrs_x_train_scaled)

# AdaBoost ?????? ??????
knn_clf = KNeighborsClassifier()

# ?????? (grid-search ???)
knn_clf.fit(updrs_x_train_scaled, updrs_y_train)

# ?????? (grid-search ???)
updrs_pred_knn_before = knn_clf.predict(updrs_x_test_scaled)

print('grid search ??? KNN ?????? ????????? ?????? y_test / ?????? y_pred: ', accuracy_score(updrs_y_test, updrs_pred_knn_before))

print('------------------ Before Test KNN ------------------------')
print(confusion_matrix(updrs_y_test, updrs_pred_knn_before))
print(classification_report(updrs_y_test, updrs_pred_knn_before, target_names=['class 0', 'class 1']))

# ??????????????? ???????????? ????????? ?????? --> ?????? ??????????????? ????????? ??????????????? ????????? ??????.
parameters = {'n_neighbors': [3, 5, 7, 9], 'leaf_size': [30, 50, 70, 90], 'p': [1, 2]}

grid_knn = GridSearchCV(knn_clf, param_grid=parameters, cv=5, refit=True)

# ?????? (grid-search ???)
grid_knn.fit(updrs_x_train_scaled, updrs_y_train)

# GridSearchCV ????????? ????????? ????????? ??????????????? ??????
scores_knn = pd.DataFrame(grid_knn.cv_results_)
scores_knn = scores_knn[['params', 'mean_test_score', 'rank_test_score',
                         'split0_test_score', 'split1_test_score', 'split2_test_score']]

print('KNN GridSearch ?????? ????????????: ', grid_knn.best_params_)
print('KNN Trees GridSearch ?????? ??????: ', grid_knn.best_score_)

knn_estimator = grid_knn.best_estimator_

# GridSearchCV??? best_estmator_ ??? ?????? ?????? ????????? ???????????? ?????? ????????? ?????? ??????
updrs_pred_knn_after = knn_estimator.predict(updrs_x_test_scaled)
print('grid-search ??? KNN test ??????????????? ?????????: {0: .4f}'.format(accuracy_score(updrs_y_test, updrs_pred_knn_after)))

print('------------------ After Test KNN ------------------------')
print(confusion_matrix(updrs_y_test, updrs_pred_knn_after))
print(classification_report(updrs_y_test, updrs_pred_knn_after, target_names=['class 0', 'class 1']))

# ---------------------------------------------------------------------------------------------------------------

# Save Model
saved_model = pickle.dumps(knn_estimator)

# Load the pickled model
clf_from_pickle = pickle.loads(saved_model)

# Use the loaded pickled model to make predictions (test data)
result_pickle = clf_from_pickle.predict(updrs_x_test_scaled)

print('KNN Saved Model POMA Accuracy: {0: .4f}'.format(accuracy_score(updrs_y_test, result_pickle)))

print('------------------ Test KNN Pickle Model ------------------------')
print(confusion_matrix(updrs_y_test, result_pickle))
print(classification_report(updrs_y_test, result_pickle, target_names=['class 0', 'class 1']))

# pickled model??? ????????? ????????? joblib ????????? ??????
joblib.dump(knn_estimator, 'knn_updrs_2class_210622.pkl')

# ---------------------------------------------------------------------------------------------------------------
