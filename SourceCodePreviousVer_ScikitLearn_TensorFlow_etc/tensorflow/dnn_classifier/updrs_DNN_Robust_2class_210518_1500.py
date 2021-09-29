import numpy as np
import pandas as pd
import tensorflow as tf
import re

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
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


# poma_2class = pd.read_csv('../../dataset/poma_dataset_210518_1500.csv')
updrs_2class = pd.read_csv('../../../dataset/updrs_dataset_210518_1500.csv')

# poma_dataset_2class = poma_2class.copy()
updrs_dataset_2class = updrs_2class.copy()

# TF2 Pipe-line에는 컬럼명에 특수문자 불가.
cols = [re.sub(r'[\W_]', "", i) for i in updrs_dataset_2class.columns]

for i in range(len(cols)):
    cols[i] = cols[i] + str(i)

updrs_dataset_2class.columns = cols

print(updrs_dataset_2class)

dftrain, dftest = train_test_split(updrs_dataset_2class, test_size=0.2,
                                   stratify=updrs_dataset_2class['updrsdanger2class0'], shuffle=True, random_state=1234)

# 'danger' 컬럼 pop -> 라벨
y_train = dftrain.pop('updrsdanger2class0')
y_test = dftest.pop('updrsdanger2class0')

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
NUMERIC_COLUMNS = updrs_dataset_2class.columns[1:]  # 첫번째 열인 'updrs_danger_3class' 제외

feature_columns = []

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

# Training and evaluation input functions.
# 이떄 반드시 모델에 입력되는 타입은 Function. (그래서 make_input_fn 메소드를 통과시켜 호출 가능한 객체로 리턴받는 것)
train_input_fn = make_input_fn(x_train_scaled_df, y_train)

# 모델 훈련 및 평가
# 단계 1. feature, hyper-parameter 지정하고 모델 초기화
# 단계 2. train_input_fn을 사용하여 훈련 데이터를 모델에 입력하고 train 함수를 사용하여 모델을 훈련.
# 단계 3. 평가 세트(dfeval DataFrame)를 사용하여 모델 성능을 평가. 예측이 y_eval 배열의 레이블과 일치하는지 확인.

# estimator using the ProximalAdagradOptimizer optimizer with regularization.
dnn_estimator = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[1024, 512, 256, 128, 64],
                                           n_classes=2, dropout=0.25, batch_norm=True)

# 모델 학습
dnn_estimator.train(train_input_fn)

# 테스트 데이터도 스케일링
x_test_scaled = rs_scaler.transform(dftest)

x_test_scaled_df = pd.DataFrame(x_test_scaled, columns=dftest.columns)

# 훈련된 DNN 모델로 테스트 데이터 셋 예측 (단, 테스트 데이터는 label 없이 feature로만 구성))
test_input_fn = make_input_fn_test(x_test_scaled_df, num_epochs=1, shuffle=True)

print('------------------ Test Predict ------------------')
pred = list(dnn_estimator.predict(test_input_fn))
pred_df = pd.DataFrame(pred)

print(pred_df)
print('------------------------------------------')
print(pred_df['class_ids'])

class_ids = pred_df['class_ids'].astype('float64')

print('------------------ Test DNN ------------------------')
print(confusion_matrix(y_test, class_ids))
print(classification_report(y_test, class_ids, target_names=['class 0', 'class 1']))

# ------------------ Test Predict ------------------
#              logits      logistic  ... all_class_ids   all_classes
# 0    [-0.008195699]   [0.4979511]  ...        [0, 1]  [b'0', b'1']
# 1     [-0.12591662]  [0.46856236]  ...        [0, 1]  [b'0', b'1']
# 2     [0.060137816]   [0.5150299]  ...        [0, 1]  [b'0', b'1']
# 3    [-0.014151824]   [0.4964621]  ...        [0, 1]  [b'0', b'1']
# 4      [0.08972878]   [0.5224171]  ...        [0, 1]  [b'0', b'1']
# ..              ...           ...  ...           ...           ...
# 295   [0.038955715]   [0.5097377]  ...        [0, 1]  [b'0', b'1']
# 296   [0.009564499]   [0.5023911]  ...        [0, 1]  [b'0', b'1']
# 297    [0.30113447]  [0.57471985]  ...        [0, 1]  [b'0', b'1']
# 298  [-0.051124338]   [0.4872217]  ...        [0, 1]  [b'0', b'1']
# 299   [0.046964955]   [0.5117391]  ...        [0, 1]  [b'0', b'1']
#
# [300 rows x 7 columns]

# ------------------------------------------
# 0      [0]
# 1      [0]
# 2      [1]
# 3      [0]
# 4      [1]
#       ...
# 295    [1]
# 296    [1]
# 297    [1]
# 298    [0]
# 299    [1]

# Name: class_ids, Length: 300, dtype: object

# ------------------ Test DNN ------------------------
# [[63 75]
#  [63 99]]
#               precision    recall  f1-score   support
#
#      class 0       0.50      0.46      0.48       138
#      class 1       0.57      0.61      0.59       162
#
#     accuracy                           0.54       300
#    macro avg       0.53      0.53      0.53       300
# weighted avg       0.54      0.54      0.54       300



