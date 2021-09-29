import numpy as np
import pandas as pd
import tensorflow as tf
import re

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler

tf.random.set_seed(210518)


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
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

    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))

        if shuffle:
            ds = ds.shuffle(1000)

        ds = ds.batch(batch_size).repeat(num_epochs)

        return ds

    return input_function


def make_input_fn_test(data_df, num_epochs=10, shuffle=True, batch_size=32):
    """
    input_function은 입력 파이프라인을 스트리밍으로 공급하는 tf.data.Dataset으로 데이터를 변환하는 방법을 명시.
    tf.data.Dataset은 데이터 프레임, CSV 형식 파일 등과 같은 여러 소스를 사용한다.

    테스트 데이터를 위한 메소드
    학습 및 검증 데이터는 {데이터 값: 라벨}의 형태였지만, 테스트에서는 라벨 없이 {데이터 값}의 dict 형태
    """

    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df)))

        if shuffle:
            ds = ds.shuffle(1000)

        ds = ds.batch(batch_size).repeat(num_epochs)

        return ds

    return input_function


poma_2class = pd.read_csv('../../../dataset/poma_dataset_210518_1500.csv')
# updrs_2class = pd.read_csv('../../dataset/updrs_dataset_210518_1500.csv')

poma_dataset_2class = poma_2class.copy()
# updrs_dataset_2class = updrs_2class.copy()

# TF2 Pipe-line에는 컬럼명에 특수문자 불가.
cols = [re.sub(r'[\W_]', "", i) for i in poma_dataset_2class.columns]

for i in range(len(cols)):
    cols[i] = cols[i] + str(i)

poma_dataset_2class.columns = cols

print(poma_dataset_2class)

dftrain, dftest = train_test_split(poma_dataset_2class, test_size=0.2,
                                   stratify=poma_dataset_2class['pomadanger2class0'], shuffle=True, random_state=1234)

# 'danger' 컬럼 pop -> 라벨
y_train = dftrain.pop('pomadanger2class0')
y_test = dftest.pop('pomadanger2class0')

print(dftrain.shape, '훈련 샘플')
print(dftest.shape, '테스트 샘플')

print(y_train.shape, '훈련 샘플 라벨')
print(y_test.shape, '테스트 샘플 라벨')

# 변형 객체 생성
rs_scaler = RobustScaler()

# 훈련데이터의 모수 분포 저장
rs_scaler.fit(dftrain)

# 훈련 데이터 스케일링
x_train_scaled = rs_scaler.transform(dftrain)

# 스케일링 된 결과 값으로 본래 값을 구할 수도 있다.
x_origin = rs_scaler.inverse_transform(x_train_scaled)

# input function의 input은 dataframe type.
x_train_scaled_df = pd.DataFrame(x_train_scaled, columns=dftrain.columns)

# 데이터 셋에서 레이블에 해당하는 위험도(danger)를 제외한 모든 열 => 수치형 열(Numeric columns)
NUMERIC_COLUMNS = poma_dataset_2class.columns[1:]  # 첫번째 열인 'poma_danger_2class' 제외한 전체 컬럼

feature_columns = []

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

train_input_fn = make_input_fn(x_train_scaled_df, y_train)

# input function 확인 해보기
ds = make_input_fn(x_train_scaled_df, y_train, batch_size=10)()

for feature_batch, label_batch in ds.take(1):
    print('특성 키:, ', list(feature_batch.keys()))
    print('"Velocityms1" 배치: ', feature_batch['Velocityms1'].numpy())
    print('레이블 배치: ', label_batch.numpy())

    #  tf.keras.layers.DenseFeatures 층을 사용하여 특정한 특성 열의 결과 확인
    Velocityms0_column = feature_columns[7]
    print(tf.keras.layers.DenseFeatures([Velocityms0_column])(feature_batch).numpy())

# Logistic Regression 모델 생성 (= tf.estimator.LinearClassifier)
linear_estimator = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=2)

# 학습 데이터로 train
linear_estimator.train(train_input_fn)

# 테스트 데이터도 스케일링
x_test_scaled = rs_scaler.transform(dftest)

x_test_scaled_df = pd.DataFrame(x_test_scaled, columns=dftest.columns)

# 훈련된 LR 모델로 테스트 데이터 셋 예측 (단, 테스트 데이터는 label 없이 feature로만 구성))
test_input_fn = make_input_fn_test(x_test_scaled_df, num_epochs=1, shuffle=True)

print('------------------ Test Predict ------------------')
pred = list(linear_estimator.predict(test_input_fn))
pred_df = pd.DataFrame(pred)

print(pred_df)
print('------------------------------------------')
print(pred_df['class_ids'])

class_ids = pred_df['class_ids'].astype('float64')

print('------------------ Test LR ------------------------')
print(confusion_matrix(y_test, class_ids))
print(classification_report(y_test, class_ids, target_names=['class 0', 'class 1']))


# ------------------ Test Predict ------------------
#             logits      logistic  ... all_class_ids   all_classes
# 0      [1.5307436]  [0.82211506]  ...        [0, 1]  [b'0', b'1']
# 1    [-0.76652825]   [0.3172306]  ...        [0, 1]  [b'0', b'1']
# 2     [-1.5396881]  [0.17658061]  ...        [0, 1]  [b'0', b'1']
# 3     [-1.8680329]  [0.13376945]  ...        [0, 1]  [b'0', b'1']
# 4    [-0.14166641]  [0.46464252]  ...        [0, 1]  [b'0', b'1']
# ..             ...           ...  ...           ...           ...
# 295   [-1.7799304]  [0.14431173]  ...        [0, 1]  [b'0', b'1']
# 296    [3.6105444]  [0.97367465]  ...        [0, 1]  [b'0', b'1']
# 297  [-0.14743042]  [0.46320903]  ...        [0, 1]  [b'0', b'1']
# 298  [-0.66533566]  [0.33954206]  ...        [0, 1]  [b'0', b'1']
# 299  [-0.17919767]   [0.4553201]  ...        [0, 1]  [b'0', b'1']
#
# [300 rows x 7 columns]

# ------------------------------------------
# 0      [1]
# 1      [0]
# 2      [0]
# 3      [0]
# 4      [0]
#       ...
# 295    [0]
# 296    [1]
# 297    [0]
# 298    [0]
# 299    [0]

# Name: class_ids, Length: 300, dtype: object

# ------------------ Test LR ------------------------
# [[67 80]
#  [77 76]]
#               precision    recall  f1-score   support
#
#      class 0       0.47      0.46      0.46       147
#      class 1       0.49      0.50      0.49       153
#
#     accuracy                           0.48       300
#    macro avg       0.48      0.48      0.48       300
# weighted avg       0.48      0.48      0.48       300


