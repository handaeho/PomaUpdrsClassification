import numpy as np
import pandas as pd
import tensorflow as tf
import re

from shutil import copytree
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow import estimator
from os import path, listdir

print('tensorflow version:', tf.__version__)

tf.compat.v1.disable_eager_execution()
tf.random.set_seed(210504)


def get_abs_directory_list(in_path):
    """입력된 경로의 (절대경로)디렉토리 목록을 가져옵니다"""
    out_paths = []
    out_ids = []
    if path.exists(in_path) and len(listdir(in_path)) >= 1:
        for id in listdir(in_path):
            abs_dir = path.join(in_path, id)
            if path.isdir(abs_dir):
                out_paths.append(abs_dir)
                out_ids.append(id)

    return out_paths, out_ids

##############################################################################################################

model_dir = './poma_dnn_test_210804'  # 모델을 저장할 디렉토리
model_name = 'poma_dnn_0001'  # 모델명
num_epochs = 20  # 에폭(입력 데이터를 몇회 순환할지)

##############################################################################################################
poma_3class = pd.read_csv('../../../dataset/real_poma_3class_dataset_210518.csv')
# updrs_3class = pd.read_csv('../../dataset/real_updrs_3class_dataset_210518.csv')

poma_dataset_3class = poma_3class.copy()
# updrs_dataset_3class = updrs_3class.copy()

# TF2 Pipe-line에는 컬럼명에 특수문자 불가.
cols = [re.sub(r'[\W_]', "", i) for i in poma_dataset_3class.columns]

for i in range(len(cols)):
    cols[i] = cols[i] + str(i)

poma_dataset_3class.columns = cols

print(poma_dataset_3class)

# 'danger' 컬럼 pop -> 라벨
labels = poma_dataset_3class.pop('pomadanger3class0').values

x_train, x_test, y_train, y_test = train_test_split(poma_dataset_3class, labels, test_size=0.2,
                                                    shuffle=True, random_state=1220)
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

# 변형 객체 생성
rs_scaler = RobustScaler()

# 훈련 데이터 스케일링
x_train_scaled = rs_scaler.fit_transform(x_train)

# 테스트 데이터의 스케일링
x_test_scaled = rs_scaler.transform(x_test)

# print(x_train_scaled)
# print(x_test_scaled)

##############################################################################################################

# 데이터 셋에서 레이블에 해당하는 위험도(danger)를 제외한 모든 열 => 수치형 열(Numeric columns)
NUMERIC_COLUMNS = poma_dataset_3class.columns

feature_columns = []

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

##############################################################################################################

def serving_input_receiver_fn():
    """Build the serving inputs."""
    inputs = {}

    for feat in poma_dataset_3class:
        inputs[feat] = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)

    print(inputs)

    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

print(serving_input_receiver_fn())

##############################################################################################################

train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(x={'x': x_train_scaled}, y=y_train,
                                                              num_epochs=num_epochs, shuffle=True)

train_spec = estimator.TrainSpec(input_fn=train_input_fn)

test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(x={'x': x_test_scaled}, y=y_test,
                                                             num_epochs=1, shuffle=False)

##############################################################################################################

# 최근 모델을 저장할 Exporter
latest_exporter = estimator.LatestExporter(name='latest_exporter',
                                           serving_input_receiver_fn=serving_input_receiver_fn)
# 가장 좋은 모델을 저장할 Exporter
best_exporter = estimator.BestExporter(name='best_exporter',
                                       serving_input_receiver_fn=serving_input_receiver_fn)

exporters = [latest_exporter, best_exporter]

eval_spec = estimator.EvalSpec(input_fn=test_input_fn, throttle_secs=10, exporters=exporters)

# dnn_estimator = tf.estimator.DNNClassifier(config=estimator.RunConfig(model_dir=model_dir),
#                                            feature_columns=[tf.feature_column.numeric_column('x', shape=[95])],
#                                            hidden_units=[1024, 512, 256], n_classes=3)

dnn_estimator = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[1024, 512, 256], n_classes=3)

tf.estimator.train_and_evaluate(dnn_estimator, train_spec=train_spec, eval_spec=eval_spec)

##############################################################################################################

print('------------------ Test Predict ------------------')
pred = list(dnn_estimator.predict(test_input_fn))
pred_df = pd.DataFrame(pred)

print(pred_df)
print('------------------------------------------')
print(pred_df['class_ids'])

class_ids = pred_df['class_ids'].astype('float64')

print('------------------ Test DNN ------------------------')
print(confusion_matrix(y_test, class_ids))
print(classification_report(y_test, class_ids, target_names=['class 0', 'class 1', 'class 2']))

export_path = dnn_estimator.export_saved_model("/daeho/PomaUpdrs/tensorflow/dnn_classifier/00001_210812_MODEL_TEST/00001_MODEL", serving_input_receiver_fn=serving_input_receiver_fn, experimental_mode=tf.estimator.ModeKeys.PREDICT)
print(export_path)

##############################################################################################################

# # 가장 좋은 모델이 저장된 디렉토리
# best_exporter_path = path.join(model_dir, 'export', 'best_exporter')
# src_paths, src_ids = get_abs_directory_list(best_exporter_path)
#
# # 서빙되고 있는 모델이 저장된 디렉토리
# serving_exporter_path = path.join(model_dir, 'export', 'serving_exporter', model_name)
# _, des_ids = get_abs_directory_list(serving_exporter_path)
#
# for idx, src_path in enumerate(src_paths):
#     # 순회
#     if src_ids[idx] not in des_ids:
#         # 신규 모델이라면
#         copytree(src_path, path.join(serving_exporter_path, src_ids[idx]))  # 복사한다
#         print(str(src_ids[idx]) + ' copy!')

#############################################################################################################

