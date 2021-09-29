import numpy as np
import pandas as pd
import tensorflow as tf
import re

from sklearn.metrics import confusion_matrix, classification_report
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

# 'danger' 컬럼 pop -> 라벨
y_train_t = dftrain_t.pop('pomadanger3class0')
y_eval = dfeval.pop('pomadanger3class0')
y_test = dftest.pop('pomadanger3class0')

print(dftrain_t.shape, '훈련 샘플')
print(dfeval.shape, '검증 샘플')
print(dftest.shape, '테스트 샘플')
# (12662, 95) 훈련 샘플
# (3166, 95) 검증 샘플
# (3958, 95) 테스트 샘플

print(y_train_t.shape, '훈련 샘플 라벨')
print(y_eval.shape, '검증 샘플 라벨')
print(y_test.shape, '테스트 샘플 라벨')
# (12662,) 훈련 샘플 라벨
# (3166,) 검증 샘플 라벨
# (3958,) 테스트 샘플 라벨

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
NUMERIC_COLUMNS = poma_dataset_3class.columns[1:]  # 첫번째 열인 'poma_danger_3class' 제외

feature_columns = []

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
# -> tf.feature_column을 이용해 모델에 맞게 데이터를 변형하거나, 데이터의 형식을 지정해줄 수 있다.
# 이러한 tf.feature_column을 통해 처리한 데이터를 Estimator(여기 참고)에 쉽게 적용하여 모델링 할 수 있다.
print(feature_columns)

# input function 확인 해보기
ds = make_input_fn(x_train_scaled_df, y_train_t, batch_size=10)()

for feature_batch, label_batch in ds.take(1):
    print('특성 키:, ', list(feature_batch.keys()))
    print('"Velocityms1" 배치: ', feature_batch['Velocityms1'].numpy())
    print('레이블 배치: ', label_batch.numpy())
    # -> 특성 키:,  ['Velocityms1', 'Cycletimes2', 'LCycletimes3', 'RCycletimes4', 'LStridelengthm5', ...]
    # "Velocityms1" 배치:  [-1.14285714 -1.02380952 -0.04761905  1.61904762 -0.88095238  0., ...]
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

# Training and evaluation input functions.
# 이떄 반드시 모델에 입력되는 타입은 Function. (그래서 make_input_fn 메소드를 통과시켜 호출 가능한 객체로 리턴받는 것)
train_input_fn = make_input_fn(x_train_scaled_df, y_train_t)
eval_input_fn = make_input_fn(x_eval_scaled_df, y_eval, num_epochs=1, shuffle=True)

# 모델 훈련 및 평가
# 단계 1. feature, hyper-parameter 지정하고 모델 초기화
# 단계 2. train_input_fn을 사용하여 훈련 데이터를 모델에 입력하고 train 함수를 사용하여 모델을 훈련.
# 단계 3. 평가 세트(dfeval DataFrame)를 사용하여 모델 성능을 평가. 예측이 y_eval 배열의 레이블과 일치하는지 확인.

gbt_estimator = tf.estimator.BoostedTreesClassifier(feature_columns=feature_columns, n_trees=300,
                                                    max_depth=5, n_batches_per_layer=32,
                                                    n_classes=3, learning_rate=0.04)
# DFC를 얻으려면 center_bias = True를 활성화해야합니다. 이렇게하면 모델이 기능을 사용하기 전에 초기 예측을 수행하게됩니다.
# (예 : 회귀에는 학습 레이블의 평균을 사용하고 교차 엔트로피 손실을 사용할 때 분류에는 로그 확률을 사용)
# 단, center_bias = True 활성화 시, n_classes > 2는 불가.

"""
그래디언트 부스팅에서 중요한 매개변수는 이전 트리의 오차를 얼마나 강하게 보정할 것인지를 제어하는 learning_rate이다.
learning_rate가 크면 트리의 오차 보정을 강하게 하기 때문에 복잡한 모델을 생성한다.
또한, n_trees 값을 키우면 앙상블에 트리가 더 많이 추가 되어 모델의 복잡도가 커지고 훈련 세트에서의 실수를 바로 잡을 기회가 더 많아진다.
과대적합을 막기 위해서최대 깊이(max_depth)를 줄이거나 사전 가지치기를 강하게 하거나 학습률(learning_rate)를 낮출 수 있다.

학습률을 낮추는 것은 테스트 세트의 성능을 조금밖에 개선 못했지만, 트리의 최대 깊이를 낮추는 것은 모델 성능 향상에 크게 기여했다.

비슷한 종류의 데이터에서 그래디언트 부스팅과 랜덤 포레스트 둘다 잘 작동된다.보통은 더 안정적인 랜덤 포레스트를 먼저 적용한다.
아무리 랜덤 포레스트가 잘 작동하더라도 예측 시간이 중요하거나 머신러닝 모델에서 마지막 성능까지 쥐어 짜야 할 때는 그레디언 부스팅이 도움이 된다.

장단점과 매개변수
그래디언트 부스팅 결정 트리는 지도학습에서 가장 강력하고 널리 사용하는 모델 중 하나이다.

장점
    다른 트리 기반 모델처럼 특성의 스케일을 조정하지 않아도 된다.
    이진(binary) 특성이나 연속적인 특성에서도 잘 동작한다.

단점
    매개변수를 잘 조정해야만 한다.
    훈련 시간이 길다.
    트리 기반 모델의 특성상 희소한 고차원 데이터에는 잘 작동하지 못한다.

그래디언트 부스팅 트리 모델의 매개변수
    n_estimators : 트리의 개수 지정. 해당 매개변수가 클수록 랜덤 포레스트와 달리 그래디언 부스팅에서는 모델이 복잡해지고 과대적합될 가능성이 높아진다.

    learning_rate : 이전 트리의 오차를 보정하는 정도. n_trees와 이 매개변수는 매우 깊게 연관되어있으며 해당 변수를 낮추면
    비슷한 복잡도의 모델을 만들기 위해서는 n_trees를 늘려서 더 많은 트리를 추가해야한다.
    일반적인 관례는 가용한 시간과 메모리 한도에서 n_estimators를 맞추고 나서 적절한 learning_rate를 찾는 것이다.

    max_depth(or max_leaf_nodes) : 트리의 복잡도를 지정.
    통상 그래디언트 부스팅모델에서는 이 매개변수를 매우 작게 설정하며 트리의 깊이가 5보다 깊어지지 않게 한다.
"""

# 지정된 수의 트리가 구축되면 모델은 학습을 중지합니다. 단계 수를 기반으로 하지 않습니다.
gbt_estimator.train(train_input_fn)

# 검증
result_eval = gbt_estimator.evaluate(eval_input_fn)

print('------------------ Eval Predict ------------------')
print(result_eval)

# 테스트 데이터도 스케일링
x_test_scaled = rs_scaler.transform(dftest)

x_test_scaled_df = pd.DataFrame(x_test_scaled, columns=dftest.columns)

# 훈련된 GBT 모델로 테스트 데이터 셋 예측 (단, 테스트 데이터는 label 없이 feature로만 구성))
test_input_fn = make_input_fn_test(x_test_scaled_df, num_epochs=1, shuffle=False)

print('------------------ Test Predict ------------------')
pred = list(gbt_estimator.predict(test_input_fn))
pred_df = pd.DataFrame(pred)

print(pred_df)
print('------------------------------------------')
print(pred_df['class_ids'])

class_ids = pred_df['class_ids'].astype('float64')

print('------------------ Test GBT ------------------------')
print(confusion_matrix(y_test, class_ids))
print(classification_report(y_test, class_ids, target_names=['class 0', 'class 1', 'class 2']))


# ------------------ Eval Predict ------------------
# {'accuracy': 0.9024005, 'average_loss': 0.29423362, 'loss': 0.29420578, 'global_step': 19768}

# ------------------ Test Predict ------------------
#                                      logits  ...         all_classes
# 0         [4.010409, 1.3950354, -3.3887007]  ...  [b'0', b'1', b'2']
# 1     [-0.056113217, 1.9091122, -0.7016428]  ...  [b'0', b'1', b'2']
# 2       [1.0050724, -0.12264744, 2.1098623]  ...  [b'0', b'1', b'2']
# 3       [2.946762, 0.15883523, -0.73303074]  ...  [b'0', b'1', b'2']
# 4      [-0.17396101, -1.6292905, 2.0577087]  ...  [b'0', b'1', b'2']
# ...                                     ...  ...                 ...
# 3953    [2.0647643, -0.43141055, -1.723591]  ...  [b'0', b'1', b'2']
# 3954    [4.351408, 0.09369871, 0.035621498]  ...  [b'0', b'1', b'2']
# 3955      [-0.8772057, 1.569899, -2.828238]  ...  [b'0', b'1', b'2']
# 3956    [0.23886016, 2.0085132, -2.8760686]  ...  [b'0', b'1', b'2']
# 3957   [-0.10928759, 0.26216194, 2.2447383]  ...  [b'0', b'1', b'2']
#
# [3958 rows x 6 columns]

# ------------------------------------------
# 0       [0]
# 1       [1]
# 2       [2]
# 3       [0]
# 4       [2]
#        ...
# 3953    [0]
# 3954    [0]
# 3955    [1]
# 3956    [1]
# 3957    [2]

# Name: class_ids, Length: 3958, dtype: object

# ------------------ Test GBT ------------------------
# [[1859   82   47]
#  [ 116 1210   14]
#  [  69   28  533]]
#               precision    recall  f1-score   support
#
#      class 0       0.91      0.94      0.92      1988
#      class 1       0.92      0.90      0.91      1340
#      class 2       0.90      0.85      0.87       630
#
#     accuracy                           0.91      3958
#    macro avg       0.91      0.89      0.90      3958
# weighted avg       0.91      0.91      0.91      3958


# Estimator Model save
serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
    tf.feature_column.make_parse_example_spec(feature_columns))

export_path = gbt_estimator.export_saved_model("new_poma_gradient_boosting_210518", serving_input_fn)

