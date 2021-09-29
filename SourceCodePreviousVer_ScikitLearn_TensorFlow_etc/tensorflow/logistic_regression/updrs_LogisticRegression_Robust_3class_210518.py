import numpy as np
import pandas as pd
import tensorflow as tf
import re

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

tf.random.set_seed(210518)


def make_input_fn(data_df, label_df, num_epochs=50, shuffle=True, batch_size=32):
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


def make_input_fn_test(data_df, num_epochs=1, shuffle=True, batch_size=32):
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


# poma_3class = pd.read_csv('../../dataset/real_poma_3class_dataset_210518.csv')
updrs_3class = pd.read_csv('../../../dataset/real_updrs_3class_dataset_210518.csv')

# poma_dataset_3class = poma_3class.copy()
updrs_dataset_3class = updrs_3class.copy()


# TF2 Pipe-line에는 컬럼명에 특수문자 불가.
cols = [re.sub(r'[\W_]', "", i) for i in updrs_dataset_3class.columns]

for i in range(len(cols)):
    cols[i] = cols[i] + str(i)

updrs_dataset_3class.columns = cols

print(updrs_dataset_3class)

dftrain, dftest = train_test_split(updrs_dataset_3class, test_size=0.2,
                                   stratify=updrs_dataset_3class['updrsdanger3class0'], shuffle=True, random_state=1220)
dftrain_t, dfeval = train_test_split(dftrain, test_size=0.2, stratify=dftrain['updrsdanger3class0'],
                                     shuffle=True, random_state=1220)

# 'danger' 컬럼 pop -> 라벨
y_train_t = dftrain_t.pop('updrsdanger3class0')
y_eval = dfeval.pop('updrsdanger3class0')
y_test = dftest.pop('updrsdanger3class0')

print(dftrain_t.shape, '훈련 샘플')
print(dftest.shape, '테스트 샘플')
print(dfeval.shape, '검증 샘플')

print(y_train_t.shape, '훈련 샘플 라벨')
print(y_test.shape, '테스트 샘플 라벨')
print(y_eval.shape, '검증 샘플 라벨')

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
x_train_scaled_df = pd.DataFrame(x_train_scaled, columns=dftrain_t.columns)
x_eval_scaled_df = pd.DataFrame(x_eval_scaled, columns=dfeval.columns)

# 데이터 셋에서 레이블에 해당하는 위험도(danger)를 제외한 모든 열 => 수치형 열(Numeric columns)
NUMERIC_COLUMNS = updrs_dataset_3class.columns[1:]  # 첫번째 열인 'updrs_danger_2class' 제외

feature_columns = []

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

train_input_fn = make_input_fn(x_train_scaled_df, y_train_t)
eval_input_fn = make_input_fn(x_eval_scaled_df, y_eval, num_epochs=1, shuffle=True)

# input function 확인 해보기
ds = make_input_fn(x_train_scaled_df, y_train_t, batch_size=10)()

for feature_batch, label_batch in ds.take(1):
    print('특성 키:, ', list(feature_batch.keys()))
    print('"Velocityms1" 배치: ', feature_batch['Velocityms1'].numpy())
    print('레이블 배치: ', label_batch.numpy())
    # -> 특성 키:,  ['Velocityms1', 'Cycletimes2', 'LCycletimes3', 'RCycletimes4', 'LStridelengthm5', ...]
    # "Velocityms1" 배치:  [ 0.12195122 -0.04878049  0.02439024 -0.53658537 -0.31707317 -0.04878049, ...]
    # 레이블 배치:  [0 1 0 1 0 1 0 1 1 0]

    #  tf.keras.layers.DenseFeatures 층을 사용하여 특정한 특성 열의 결과 확인
    Velocityms0_column = feature_columns[7]
    print(tf.keras.layers.DenseFeatures([Velocityms0_column])(feature_batch).numpy())
    # [[-0.55023366]
    #  [-0.5       ]
    #  [ 0.18457943]
    #  ...
    #  [ 0.5186916 ]
    #  [ 1.7558411 ]
    #  [-1.1810747 ]]

# Logistic Regression 모델 생성 (= tf.estimator.LinearClassifier)
linear_estimator = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=3)

# 학습 데이터로 train
linear_estimator.train(train_input_fn)

# 검증 데이터로 eval
result_eval = linear_estimator.evaluate(eval_input_fn)

print('------------------ Eval Predict ------------------')
print(result_eval)

# 테스트 데이터도 스케일링
x_test_scaled = rs_scaler.transform(dftest)

x_test_scaled_df = pd.DataFrame(x_test_scaled, columns=dftest.columns)

# 훈련된 LR 모델로 테스트 데이터 셋 예측
test_input_fn = make_input_fn_test(x_test_scaled_df, num_epochs=1, shuffle=False)

print('------------------ Test Predict ------------------')
pred = list(linear_estimator.predict(test_input_fn))
pred_df = pd.DataFrame(pred)

print(pred_df)
print('------------------------------------------')
print(pred_df['class_ids'])

class_ids = pred_df['class_ids'].astype('float64')

print('------------------ Test LR ------------------------')
print(confusion_matrix(y_test, class_ids))
print(classification_report(y_test, class_ids, target_names=['class 0', 'class 1', 'class 2']))

# ------------------ Eval Predict ------------------
# {'accuracy': 0.6648768, 'average_loss': 0.7705802, 'loss': 0.77060884, 'global_step': 19800}

# ------------------ Test Predict ------------------
#                                     logits  ...         all_classes
# 0      [1.4306669, 0.46058518, -2.1679156]  ...  [b'0', b'1', b'2']
# 1       [0.9343993, 0.7475976, -1.8272858]  ...  [b'0', b'1', b'2']
# 2       [0.6923968, 0.3734861, -1.2491717]  ...  [b'0', b'1', b'2']
# 3      [0.9530855, 0.43691567, -1.5023692]  ...  [b'0', b'1', b'2']
# 4      [0.5851964, 0.16491053, -0.8875871]  ...  [b'0', b'1', b'2']
# ...                                    ...  ...                 ...
# 3953   [-1.837769, 1.8294225, -0.23705548]  ...  [b'0', b'1', b'2']
# 3954    [1.1826171, 0.8046584, -1.8824031]  ...  [b'0', b'1', b'2']
# 3955  [1.0160367, 0.052212477, -1.2804282]  ...  [b'0', b'1', b'2']
# 3956    [0.91476464, 0.665315, -2.1453364]  ...  [b'0', b'1', b'2']
# 3957     [0.7209409, 0.900298, -2.0057106]  ...  [b'0', b'1', b'2']
#
# [3958 rows x 6 columns]

# ------------------------------------------
# 0       [0]
# 1       [0]
# 2       [0]
# 3       [0]
# 4       [0]
#        ...
# 3953    [1]
# 3954    [0]
# 3955    [0]
# 3956    [0]
# 3957    [1]

# Name: class_ids, Length: 3958, dtype: object

# ------------------ Test LR ------------------------
# [[1439  308   65]
#  [ 513  893  122]
#  [ 149  124  345]]
#               precision    recall  f1-score   support
#
#      class 0       0.68      0.79      0.74      1812
#      class 1       0.67      0.58      0.63      1528
#      class 2       0.65      0.56      0.60       618
#
#     accuracy                           0.68      3958
#    macro avg       0.67      0.65      0.65      3958
# weighted avg       0.67      0.68      0.67      3958


# Estimator Model save
serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
    tf.feature_column.make_parse_example_spec(feature_columns))

export_path = linear_estimator.export_saved_model("new_updrs_logistic_regression_model_210518", serving_input_fn)
