import numpy as np
import pandas as pd
import tensorflow as tf
import re

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.python.keras.utils.np_utils import to_categorical

tf.random.set_seed(1234)


def make_input(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    """
    input_function은 입력 파이프라인을 스트리밍으로 공급하는 tf.data.Dataset으로 데이터를 변환하는 방법을 명시.
    tf.data.Dataset은 데이터 프레임, CSV 형식 파일 등과 같은 여러 소스를 사용한다.

    :param data_df: dateset
    :param label_df: dataset의 라벨(타겟)
    :param num_epochs: 반복 횟수
    :param shuffle: 셔플 (True, False)
    :param batch_size: 배치 크기
    :return:
    """
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(data_df))

    ds = ds.batch(batch_size).repeat(num_epochs)

    return ds


def make_input_test(data_df, num_epochs=10, shuffle=True, batch_size=32):
    """
    input_function은 입력 파이프라인을 스트리밍으로 공급하는 tf.data.Dataset으로 데이터를 변환하는 방법을 명시.
    tf.data.Dataset은 데이터 프레임, CSV 형식 파일 등과 같은 여러 소스를 사용한다.

    테스트 데이터를 위한 메소드
    학습 및 검증 데이터는 {데이터 값: 라벨}의 형태였지만, 테스트에서는 라벨 없이 {데이터 값}의 dict 형태
    """
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df)))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(data_df))

    ds = ds.batch(batch_size).repeat(num_epochs)

    return ds


# poma_3class = pd.read_csv('../../dataset/real_poma_3class_dataset_210504.csv')
updrs_3class = pd.read_csv('../../dataset/real_updrs_3class_dataset_210504.csv')

# poma_dataset = poma_3class.copy()
updrs_dataset = updrs_3class.copy()

# TF2 Pipe-line에는 컬럼명에 특수문자 불가.
cols = [re.sub(r'[\W_]', "", i) for i in updrs_dataset.columns]

for i in range(len(cols)):
    cols[i] = cols[i] + str(i)

updrs_dataset.columns = cols

print(updrs_dataset)

dftrain, dftest = train_test_split(updrs_dataset, test_size=0.2, shuffle=True, random_state=1234)
dftrain_t, dfeval = train_test_split(dftrain, test_size=0.2, shuffle=True, random_state=1234)

# 'danger' 컬럼 pop -> 라벨
y_train_t = dftrain_t.pop('updrsdanger3class0')
y_eval = dfeval.pop('updrsdanger3class0')
y_test = dftest.pop('updrsdanger3class0')

print(len(updrs_dataset[updrs_dataset['updrsdanger3class0'] == 0]))    # 1929
print(len(updrs_dataset[updrs_dataset['updrsdanger3class0'] == 1]))    # 609
print(len(updrs_dataset[updrs_dataset['updrsdanger3class0'] == 2]))    # 654

print(len(y_train_t[y_train_t == 0]), 'train label 0')    # 1229 train label 0
print(len(y_train_t[y_train_t == 1]), 'train label 1')    # 401 train label 1
print(len(y_train_t[y_train_t == 2]), 'train label 2')    # 412 train label 2

print(len(y_eval[y_eval == 0]), 'eval label 0')    # 321 eval label 0
print(len(y_eval[y_eval == 1]), 'eval label 1')    # 81 eval label 1
print(len(y_eval[y_eval == 2]), 'eval label 2')    # 109 eval label 2

print(len(y_test[y_test == 0]), 'test label 0')    # 379 test label 0
print(len(y_test[y_test == 1]), 'test label 1')    # 127 test label 1
print(len(y_test[y_test == 2]), 'test label 2')    # 133 test label 2

print(dftrain_t.shape, '훈련 샘플')
print(dftest.shape, '테스트 샘플')
print(dfeval.shape, '검증 샘플')
# (2042, 95) 훈련 샘플
# (639, 95) 테스트 샘플
# (511, 95) 검증 샘플

print(y_train_t.shape, '훈련 샘플 라벨')
print(y_test.shape, '테스트 샘플 라벨')
print(y_eval.shape, '검증 샘플 라벨')
# (2042,) 훈련 샘플 라벨
# (639,) 테스트 샘플 라벨
# (511,) 검증 샘플 라벨

# Multi Class Classification 이니까 One-Hot Encoding
y_train_t = to_categorical(y_train_t, 3)
y_eval = to_categorical(y_eval, 3)
y_test = to_categorical(y_test, 3)

# 변형 객체 생성
rs_scaler = RobustScaler()

# 훈련데이터의 모수 분포 저장
rs_scaler.fit(dftrain_t)

# 훈련 데이터 스케일링
x_train_scaled = rs_scaler.transform(dftrain_t)

# 검증 데이터의 스케일링
x_eval_scaled = rs_scaler.transform(dfeval)

# 스케일링 된 결과 값으로 본래 값을 구할 수도 있다.
x_origin = rs_scaler.inverse_transform(x_train_scaled)

# input function의 input은 dataframe type.
x_train_scaled = pd.DataFrame(x_train_scaled, columns=dftrain_t.columns)
x_eval_scaled = pd.DataFrame(x_eval_scaled, columns=dfeval.columns)

# Training and evaluation input functions.
train_input_fn = make_input(x_train_scaled, y_train_t)
eval_input_fn = make_input(x_eval_scaled, y_eval, num_epochs=1, shuffle=True)

# input function 확인 해보기
ds = make_input(x_train_scaled, y_train_t, batch_size=32)

for feature_batch, label_batch in ds.take(1):
    print('전체 특성:', list(feature_batch.keys()))
    print('특성의 배치:', feature_batch['Velocityms1'])
    print('타깃의 배치:', label_batch)

# 데이터 셋에서 레이블에 해당하는 위험도(danger)를 제외한 모든 열 => 수치형 열(Numeric columns)
NUMERIC_COLUMNS = updrs_dataset.columns[1:]  # 첫번째 열인 'poma_danger_2class' 제외

feature_columns_list = []

for feature_name in NUMERIC_COLUMNS:
    feature_columns_list.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
# -> tf.feature_column을 이용해 모델에 맞게 데이터를 변형하거나, 데이터의 형식을 지정해줄 수 있다.
# 이러한 tf.feature_column을 통해 처리한 데이터를 Estimator에 쉽게 적용하여 모델링 할 수 있다.

# The added layer must be an instance of class Layer. 따라서 DenseFeatures layer로 만들어준다.
feature_layer = tf.keras.layers.DenseFeatures(feature_columns_list)

# MLP Model - he_normal, he_uniform /  glorot_normal, glorot_uniform
# 다층 퍼셉트론은 쭉 늘어놓은 1차원 벡터와 같은 형태의 데이터만 받아들일 수 있다.

# Label 개수
num_classes = 3

model = tf.keras.Sequential([
    feature_layer,
    tf.keras.layers.Dense(1024, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2()),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(512, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2()),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(256, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2()),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(128, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2()),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
# 그나저나 배치 정규화를 사용하면 드랍아웃은 필요없다?

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_input_fn, validation_data=eval_input_fn, epochs=20)

# eval set으로 평가
eval_result = model.evaluate(eval_input_fn, return_dict=True)
print('------------------ Eval MLP ------------------------')
print(eval_result)

# 테스트 데이터도 스케일링 (단, 테스트 데이터는 label이 없는 dict (feature만 존재))
x_test_scaled = rs_scaler.transform(dftest)

x_test_scaled = pd.DataFrame(x_test_scaled, columns=dftest.columns)

test_ds = make_input_test(x_test_scaled, shuffle=True, num_epochs=1)

# 테스트 데이터 예측
pred_test = model.predict(test_ds)

pred_df = pd.DataFrame(pred_test)
print(pred_df)

# 각 행의 최대값을 찾고, 그 컬럼명을 저장 -> 예측한 label
pred_class = pred_df.idxmax(axis=1).astype('float64').values
# print(pred_class)

# One-Hot Encoding 된 y_test label에 argmax를 사용해 원래 label 찾음.
y_test_argmax = y_test.argmax(axis=1)
# print(y_test_argmax)

print('------------------ Test MLP ------------------------')
print(confusion_matrix(y_test_argmax, pred_class))
print(classification_report(y_test_argmax, pred_class, target_names=['class 0', 'class 1', 'class 2']))

# ------------------ Eval MLP ------------------------
# {'loss': 0.446909636259079, 'accuracy': 0.8881869912147522}

#              0         1         2
# 0     0.402041  0.589481  0.008478
# 1     0.361089  0.328010  0.310901
# 2     0.924604  0.060847  0.014549
# 3     0.835642  0.119486  0.044871
# 4     0.765539  0.139811  0.094650
# ...        ...       ...       ...
# 3953  0.654764  0.342944  0.002292
# 3954  0.923896  0.067400  0.008705
# 3955  0.212982  0.783871  0.003147
# 3956  0.944904  0.046883  0.008214
# 3957  0.289769  0.707211  0.003020
#
# [3958 rows x 3 columns]

# ------------------ Test MLP ------------------------
# [[933 641 254]
#  [765 536 235]
#  [312 197  85]]

#               precision    recall  f1-score   support
#
#      class 0       0.46      0.51      0.49      1828
#      class 1       0.39      0.35      0.37      1536
#      class 2       0.15      0.14      0.15       594
#
#     accuracy                           0.39      3958
#    macro avg       0.33      0.33      0.33      3958
# weighted avg       0.39      0.39      0.39      3958


