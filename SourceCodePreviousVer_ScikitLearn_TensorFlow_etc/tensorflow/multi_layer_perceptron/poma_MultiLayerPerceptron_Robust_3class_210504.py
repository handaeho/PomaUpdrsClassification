import numpy as np
import pandas as pd
import tensorflow as tf
import re

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.python.keras.utils.np_utils import to_categorical

tf.random.set_seed(1234)


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

dftrain, dftest = train_test_split(poma_dataset_3class, test_size=0.2,
                                   stratify=poma_dataset_3class['pomadanger3class0'], shuffle=True, random_state=1220)
dftrain_t, dfeval = train_test_split(dftrain, test_size=0.2,
                                     stratify=dftrain['pomadanger3class0'], shuffle=True, random_state=1220)

dftrain = dftrain.astype(np.float32)
dftrain_t = dftrain_t.astype(np.float32)
dfeval = dfeval.astype(np.float32)
dftest = dftest.astype(np.float32)

# 'danger' 컬럼 pop -> 라벨
y_train_t = dftrain_t.pop('pomadanger3class0')
y_eval = dfeval.pop('pomadanger3class0')
y_test = dftest.pop('pomadanger3class0')

# print(dftrain_t.shape, '훈련 샘플')
# print(dftest.shape, '테스트 샘플')
# print(dfeval.shape, '검증 샘플')
#
# print(y_train_t.shape, '훈련 샘플 라벨')
# print(y_test.shape, '테스트 샘플 라벨')
# print(y_eval.shape, '검증 샘플 라벨')

# Multi Class Classification 이니까 One-Hot Encoding
y_train_t = to_categorical(y_train_t, 3)
y_eval = to_categorical(y_eval, 3)
y_test = to_categorical(y_test, 3)

# 변형 객체 생성
rs_scaler = RobustScaler()

# 훈련데이터의 모수 분포 저장
rs_scaler.fit(dftrain_t)

# 훈련 데이터 스케일링
x_train_scaled = np.array(rs_scaler.transform(dftrain_t))

# 검증 데이터의 스케일링
x_eval_scaled = np.array(rs_scaler.transform(dfeval))

# 스케일링 된 결과 값으로 본래 값을 구할 수도 있다.
x_origin = rs_scaler.inverse_transform(x_train_scaled)

# MLP Model - he_normal, he_uniform /  glorot_normal, glorot_uniform

# 다층 퍼셉트론은 쭉 늘어놓은 1차원 벡터와 같은 형태의 데이터만 받아들일 수 있다.

print(x_train_scaled, np.shape(x_train_scaled))
print()
print(x_train_scaled, np.shape(x_eval_scaled))
print()
print(y_train_t, np.shape(y_train_t))
print()
print(y_eval, np.shape(y_eval))

# Label 개수
num_classes = 3

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(), activation='relu', input_shape=(95, )),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(num_classes, activation='softmax'),
])
# 그나저나 배치 정규화를 사용하면 드랍아웃은 필요없다?

model.summary()

model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=x_train_scaled, y=y_train_t, validation_data=x_eval_scaled, epochs=50)

# eval set으로 평가
eval_result = model.evaluate(x=x_eval_scaled, y=y_eval, return_dict=True)
print('------------------ Eval MLP ------------------------')
print(eval_result)

# 테스트 데이터도 스케일링
x_test_scaled = np.array(rs_scaler.transform(dftest))

# 테스트 데이터 예측
pred_test = model.predict(x=x_test_scaled)

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
# {'loss': 1.130769968032837, 'accuracy': 0.9428300857543945}

#              0         1         2
# 0     0.941983  0.000007  0.058010
# 1     0.000399  0.000195  0.999406
# 2     0.000181  0.998103  0.001716
# 3     0.998784  0.000196  0.001020
# 4     0.011518  0.986466  0.002016
# ...        ...       ...       ...
# 3953  0.999784  0.000079  0.000137
# 3954  0.059013  0.924798  0.016189
# 3955  0.002146  0.000232  0.997622
# 3956  0.999482  0.000468  0.000050
# 3957  0.999833  0.000002  0.000165
#
# [3958 rows x 3 columns]

# ------------------ Test MLP ------------------------
# [[1012  682  294]
#  [ 670  455  215]
#  [ 334  196  100]]

#               precision    recall  f1-score   support
#
#      class 0       0.50      0.51      0.51      1988
#      class 1       0.34      0.34      0.34      1340
#      class 2       0.16      0.16      0.16       630
#
#     accuracy                           0.40      3958
#    macro avg       0.34      0.34      0.34      3958
# weighted avg       0.39      0.40      0.39      3958


