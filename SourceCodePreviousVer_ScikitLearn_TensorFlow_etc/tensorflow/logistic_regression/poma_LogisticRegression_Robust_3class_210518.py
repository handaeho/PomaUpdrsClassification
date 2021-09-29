import numpy as np
import pandas as pd
import tensorflow as tf
import re

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
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
dftrain_t, dfeval = train_test_split(dftrain, test_size=0.2, stratify=dftrain['pomadanger3class0'],
                                     shuffle=True, random_state=1220)

# 'danger' 컬럼 pop -> 라벨
y_train_t = dftrain_t.pop('pomadanger3class0')
y_eval = dfeval.pop('pomadanger3class0')
y_test = dftest.pop('pomadanger3class0')

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
NUMERIC_COLUMNS = poma_dataset_3class.columns[1:]  # 첫번째 열인 'poma_danger_2class' 제외한 전체 컬럼

feature_columns = []

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
# -> tf.feature_column을 이용해 모델에 맞게 데이터를 변형하거나, 데이터의 형식을 지정해줄 수 있다.
# 이러한 tf.feature_column을 통해 처리한 데이터를 Estimator(여기 참고)에 쉽게 적용하여 모델링 할 수 있다.
print(feature_columns)

train_input_fn = make_input_fn(x_train_scaled_df, y_train_t)
eval_input_fn = make_input_fn(x_eval_scaled_df, y_eval, num_epochs=1, shuffle=True)

# input function 확인 해보기
ds = make_input_fn(x_train_scaled_df, y_train_t, batch_size=10)()

for feature_batch, label_batch in ds.take(1):
    print('특성 키: ', list(feature_batch.keys()))
    print('"Velocityms1" 배치: ', feature_batch['Velocityms1'].numpy())
    print('레이블 배치: ', label_batch.numpy())
    # -> 특성 키:  ['Velocityms1', 'Cycletimes2', 'LCycletimes3', 'RCycletimes4', 'LStridelengthm5', ...
    # "Velocityms1" 배치:  [-1.14285714 -1.02380952 -0.04761905  1.61904762 -0.88095238, ...]
    # 레이블 배치:  [1 0 0 0 1 1 1 0 0 0]

    #  tf.keras.layers.DenseFeatures 층을 사용하여 특정한 특성 열의 결과 확인
    Velocityms0_column = feature_columns[7]
    print(tf.keras.layers.DenseFeatures([Velocityms0_column])(feature_batch).numpy())
    # [[ 0.9661215 ]
    #  [-0.64953274]
    #  [-0.60046726]
    #  ...
    #  [-2.109813  ]
    #  [-0.64953274]
    #  [-0.11799066]]

""" Logistic Regression(=Linear Classifier) 에서는 딱히 조정할 파라미터가 없다? """
# Logistic Regression 모델 생성 (= tf.estimator.LinearClassifier)
linear_estimator = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=3)

# 학습 데이터로 train
linear_estimator.train(train_input_fn)

# 검증 데이터로 eval
result_eval = linear_estimator.evaluate(eval_input_fn)

print('------------------ Eval Predict ------------------')
print(result_eval)

"""
>> tf.feature_column.crossed_column([컬럼 1, 컬럼 2])
    = 단일 feature만 학습시키지 않고, 여러 feture를 조합해서 사용할 수도 있다.
    서로 다른 특성 조합들 간의 차이 학습을 위해 모델에 교차 특성 열을 추가할 수 있음. pandas에서 groupby를 통한 feature 생성이라고 생각하면 쉽다.
"""

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
print(classification_report(y_test, class_ids, target_names=['class 0', 'class 1', 'class 2']))

# ------------------ Eval Predict ------------------
# {'accuracy': 0.70972836, 'average_loss': 0.70272815, 'loss': 0.70263004, 'global_step': 19800}

# ------------------ Test Predict ------------------
#                                      logits  ...         all_classes
# 0         [42.021095, 37.846886, 37.957798]  ...  [b'0', b'1', b'2']
# 1            [39.27142, 41.58747, 36.65627]  ...  [b'0', b'1', b'2']
# 2         [40.913044, 38.998615, 38.801357]  ...  [b'0', b'1', b'2']
# 3          [2.02416, -1.154739, -1.0578885]  ...  [b'0', b'1', b'2']
# 4         [-1.8548483, 1.578695, 0.5865248]  ...  [b'0', b'1', b'2']
# ...                                     ...  ...                 ...
# 3953  [-0.69237113, 0.5035757, -0.32144535]  ...  [b'0', b'1', b'2']
# 3954  [1.9021357, -0.028448343, -1.3436081]  ...  [b'0', b'1', b'2']
# 3955    [0.8515538, -0.5331311, -0.5014767]  ...  [b'0', b'1', b'2']
# 3956   [1.0496148, -0.17741293, -1.6654783]  ...  [b'0', b'1', b'2']
# 3957    [1.6207349, 0.10379365, -1.9154596]  ...  [b'0', b'1', b'2']
#
# [3958 rows x 6 columns]

# ------------------------------------------
# 0       [0]
# 1       [1]
# 2       [0]
# 3       [0]
# 4       [1]
#        ...
# 3953    [1]
# 3954    [0]
# 3955    [0]
# 3956    [0]
# 3957    [0]

# Name: class_ids, Length: 3958, dtype: object

# ------------------ Test LR ------------------------
# [[1117  603  268]
#  [ 758  402  180]
#  [ 360  162  108]]
#               precision    recall  f1-score   support
#
#      class 0       0.50      0.56      0.53      1988
#      class 1       0.34      0.30      0.32      1340
#      class 2       0.19      0.17      0.18       630
#
#     accuracy                           0.41      3958
#    macro avg       0.35      0.34      0.34      3958
# weighted avg       0.40      0.41      0.40      3958
#



# Estimator Model save
serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
    tf.feature_column.make_parse_example_spec(feature_columns))

export_path = linear_estimator.export_saved_model("new_poma_logistic_regression_model_210518", serving_input_fn)
