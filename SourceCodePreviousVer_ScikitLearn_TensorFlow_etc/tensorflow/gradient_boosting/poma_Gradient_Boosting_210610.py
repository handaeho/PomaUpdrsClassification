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

""" DNN - Linear Model Re-Make """

print('tensorflow version:', tf.__version__)
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


def serving_input_receiver_fn():
    """Build the serving inputs."""
    # The outer dimension (None) allows us to batch up inputs for efficiency.
    # However, it also means that if we want a prediction
    # for a single instance, we'll need to wrap it in an outer list.

    receiver_tensors = {
        'Velocityms1': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'Cycletimes2': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'LCycletimes3': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'RCycletimes4': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'LStridelengthm5': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "RStridelengthm6": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "LStrideperminStridem7": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "RStrideperminstridem8": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "LFootvelms9": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "RFootvelms10": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "Lsteptimes11": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "Rsteptimes12": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "LStepperminstepm13": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "Rstepperminstepm14": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "LStancetimes15": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "RStancetimes16": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "Lswingtimes17": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "RSwingtimes18": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "DLSTtimes19": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "DLSTInitialtimes20": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "DLSTTerminaltimes21": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "LTotal22": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "LIn23": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "Lout24": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "Lfront25": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "Lback26": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L127": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L228": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L329": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L430": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L531": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L632": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L733": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L834": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "RTotal35": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "RIn36": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "Rout37": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "Rfront38": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "Rback39": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R140": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R241": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R342": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R443": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R544": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R645": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R746": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R847": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L1BalanceTime48": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L249": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L350": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L451": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L552": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L653": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L754": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L855": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R156": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R257": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R358": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R459": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R560": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R661": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R762": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R863": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L11Sequence64": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L12Sequence65": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L2166": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L2267": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L3168": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L3269": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L4170": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L4271": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L5172": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L5273": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L6174": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L6275": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L7176": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L7277": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L8178": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "L8279": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R1180": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R1281": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R2182": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R2283": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R3184": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R3285": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R4186": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R4287": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R5188": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R5289": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R6190": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R6291": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R7192": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R7293": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R8194": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        "R8295": tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
    }

    features = {
        'x': tf.concat([
            receiver_tensors['Velocityms1'],
            receiver_tensors['Cycletimes2'],
            receiver_tensors['LCycletimes3'],
            receiver_tensors['RCycletimes4'],
            receiver_tensors['LStridelengthm5'],
            receiver_tensors['RStridelengthm6'],
            receiver_tensors['LStrideperminStridem7'],
            receiver_tensors['RStrideperminstridem8'],
            receiver_tensors['LFootvelms9'],
            receiver_tensors['RFootvelms10'],
            receiver_tensors['Lsteptimes11'],
            receiver_tensors['Rsteptimes12'],
            receiver_tensors['LStepperminstepm13'],
            receiver_tensors['Rstepperminstepm14'],
            receiver_tensors['LStancetimes15'],
            receiver_tensors['RStancetimes16'],
            receiver_tensors['Lswingtimes17'],
            receiver_tensors['RSwingtimes18'],
            receiver_tensors['DLSTtimes19'],
            receiver_tensors['DLSTInitialtimes20'],
            receiver_tensors['DLSTTerminaltimes21'],
            receiver_tensors['LTotal22'],
            receiver_tensors['LIn23'],
            receiver_tensors['Lout24'],
            receiver_tensors['Lfront25'],
            receiver_tensors['Lback26'],
            receiver_tensors['L127'],
            receiver_tensors['L228'],
            receiver_tensors['L329'],
            receiver_tensors['L430'],
            receiver_tensors['L531'],
            receiver_tensors['L632'],
            receiver_tensors['L733'],
            receiver_tensors['L834'],
            receiver_tensors['RTotal35'],
            receiver_tensors['RIn36'],
            receiver_tensors['Rout37'],
            receiver_tensors['Rfront38'],
            receiver_tensors['Rback39'],
            receiver_tensors['R140'],
            receiver_tensors['R241'],
            receiver_tensors['R342'],
            receiver_tensors['R443'],
            receiver_tensors['R544'],
            receiver_tensors['R645'],
            receiver_tensors['R746'],
            receiver_tensors['R847'],
            receiver_tensors['L1BalanceTime48'],
            receiver_tensors['L249'],
            receiver_tensors['L350'],
            receiver_tensors['L451'],
            receiver_tensors['L552'],
            receiver_tensors['L653'],
            receiver_tensors['L754'],
            receiver_tensors['L855'],
            receiver_tensors['R156'],
            receiver_tensors['R257'],
            receiver_tensors['R358'],
            receiver_tensors['R459'],
            receiver_tensors['R560'],
            receiver_tensors['R661'],
            receiver_tensors['R762'],
            receiver_tensors['R863'],
            receiver_tensors['L11Sequence64'],
            receiver_tensors['L12Sequence65'],
            receiver_tensors['L2166'],
            receiver_tensors['L2267'],
            receiver_tensors['L3168'],
            receiver_tensors['L3269'],
            receiver_tensors['L4170'],
            receiver_tensors['L4271'],
            receiver_tensors['L5172'],
            receiver_tensors['L5273'],
            receiver_tensors['L6174'],
            receiver_tensors['L6275'],
            receiver_tensors['L7176'],
            receiver_tensors['L7277'],
            receiver_tensors['L8178'],
            receiver_tensors['L8279'],
            receiver_tensors['R1180'],
            receiver_tensors['R1281'],
            receiver_tensors['R2182'],
            receiver_tensors['R2283'],
            receiver_tensors['R3184'],
            receiver_tensors['R3285'],
            receiver_tensors['R4186'],
            receiver_tensors['R4287'],
            receiver_tensors['R5188'],
            receiver_tensors['R5289'],
            receiver_tensors['R6190'],
            receiver_tensors['R6291'],
            receiver_tensors['R7192'],
            receiver_tensors['R7293'],
            receiver_tensors['R8194'],
            receiver_tensors['R8295']
        ], axis=1)
    }

    return tf.estimator.export.ServingInputReceiver(receiver_tensors=receiver_tensors, features=features)


##############################################################################################################

model_dir = './poma_gradient_boosting_test_210610/'  # 모델을 저장할 디렉토리
model_name = 'poma_gradient_boosting_0001'  # 모델명
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

gbt_estimator = tf.estimator.BoostedTreesClassifier(config=estimator.RunConfig(model_dir=model_dir),
                                                    feature_columns=[tf.feature_column.numeric_column('x', shape=[95])],
                                                    n_trees=300, max_depth=5, n_batches_per_layer=32,
                                                    n_classes=3, learning_rate=0.04)

tf.estimator.train_and_evaluate(gbt_estimator, train_spec=train_spec, eval_spec=eval_spec)

##############################################################################################################

print('------------------ Test Predict ------------------')
pred = list(gbt_estimator.predict(test_input_fn))
pred_df = pd.DataFrame(pred)

print(pred_df)
print('------------------------------------------')
print(pred_df['class_ids'])

class_ids = pred_df['class_ids'].astype('float64')

print('------------------ Test DNN ------------------------')
print(confusion_matrix(y_test, class_ids))
print(classification_report(y_test, class_ids, target_names=['class 0', 'class 1', 'class 2']))

##############################################################################################################

# 가장 좋은 모델이 저장된 디렉토리
best_exporter_path = path.join(model_dir, 'export', 'best_exporter')
src_paths, src_ids = get_abs_directory_list(best_exporter_path)

# 서빙되고 있는 모델이 저장된 디렉토리
serving_exporter_path = path.join(model_dir, 'export', 'serving_exporter', model_name)
_, des_ids = get_abs_directory_list(serving_exporter_path)

for idx, src_path in enumerate(src_paths):
    # 순회
    if src_ids[idx] not in des_ids:
        # 신규 모델이라면
        copytree(src_path, path.join(serving_exporter_path, src_ids[idx]))
        # 복사한다
        print(str(src_ids[idx]) + ' copy!')


# ------------------ Test DNN ------------------------
# [[1869   59   34]
#  [  81 1294   13]
#  [  42   20  546]]
#               precision    recall  f1-score   support
#
#      class 0       0.94      0.95      0.95      1962
#      class 1       0.94      0.93      0.94      1388
#      class 2       0.92      0.90      0.91       608
#
#     accuracy                           0.94      3958
#    macro avg       0.93      0.93      0.93      3958
# weighted avg       0.94      0.94      0.94      3958
#
# 1623222071 copy!

