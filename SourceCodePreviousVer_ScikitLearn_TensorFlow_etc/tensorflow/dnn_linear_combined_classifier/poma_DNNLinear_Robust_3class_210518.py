import numpy as np
import pandas as pd
import tensorflow as tf
import re

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

tf.random.set_seed(210518)

"""
Deep Neural Network & Linear Combined Model
    = 넓고 (linear model 특성) + 깊은 (DNN model 특성) 모델.

    tf.estimator.DNNLinearCombinedClassifier(
        model_dir=None, linear_feature_columns=None, linear_optimizer='Ftrl', dnn_feature_columns=None, 
        dnn_optimizer='Adagrad', dnn_hidden_units=None, dnn_activation_fn=tf.nn.relu, dnn_dropout=None,
        n_classes=2, weight_column=None, label_vocabulary=None, config=None, warm_start_from=None,
        loss_reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE, batch_norm=False, linear_sparse_combiner='sum'
    )

    - model_dir	
        모델 매개 변수, 그래프 등을 저장하기위한 디렉토리. 이전에 저장된 모델을 계속 훈련하기 위해 디렉토리에서 추정기로 체크 포인트를 로드하는 데 사용할 수도 있습니다.
    - linear_feature_columns	
        모델의 선형 부분에서 사용하는 모든 특성 열을 포함하는 반복 가능 객체입니다. 집합의 모든 항목은 FeatureColumn에서 파생 된 클래스의 인스턴스 여야합니다.
    - linear_optimizer	    
        모델의 선형 부분에 그라디언트(기울기)를 적용하는 데 사용되는 tf.keras.optimizers.* 의 인스턴스입니다. 
        문자열 ('Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD' 중 하나) 이거나 호출 가능 객체일 수도 있습니다.
        기본값은 FTRL 최적화 프로그램입니다.
    - dnn_feature_columns	
        모델의 딥 부분에서 사용하는 모든 특성 열을 포함하는 반복 가능 항목입니다. 
        집합의 모든 항목은 FeatureColumn 에서 파생 된 클래스의 인스턴스 여야합니다.
    - dnn_optimizer	
        모델의 깊은 부분에 그라데이션을 적용하는 데 사용되는 tf.keras.optimizers.* 의 인스턴스입니다. 
        문자열 ( 'Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD' 중 하나) 이거나 호출 가능 객체일 수도 있습니다. 기본값은 Adagrad 최적화 프로그램입니다.
    - dnn_hidden_units	
        레이어 별 은닉 유닛 목록. 모든 레이어가 완전히 연결되어 있습니다.
    - dnn_activation_fn	
        각 레이어에 적용되는 활성화 기능. None 이면 tf.nn.relu.
    - dnn_dropout	
        None이 아닌 경우 주어진 좌표를 삭제할 확률입니다.
    - n_classes	
        레이블 클래스 수입니다. 기본값은 2, 즉 이진 분류입니다. 1보다 커야합니다.
    - weight_column	
         가중치를 나타내는 특성 열을 정의하는 tf.feature_column.numeric_column에 의해 생성 된 문자열 또는 _NumericColumn. 
         훈련 중에 weight을 줄이거나 사례를 늘리는 데 사용됩니다. 예제의 손실이 곱해집니다. 문자열인 경우 기능에서 가중치 텐서를 가져 오는 키로 사용됩니다. 
         _NumericColumn이면 raw tensor를 weight_column.key 키로 가져온 다음 weight_column.normalizer_fn을 적용하여 가중치 텐서를 얻습니다.
    - label_vocabulary	
        문자열 목록은 가능한 레이블 값을 나타냅니다. 주어진 경우 레이블은 문자열 유형이어야 하며 label_vocabulary에 값이 있어야합니다.
        지정하지 않으면 레이블이 이미 n_classes = 2의 경우 [0, 1] 내에서 정수 또는 부동 소수점으로 인코딩되고 
        n_classes > 2의 경우 {0, 1, ..., n_classes-1}의 정수 값으로 인코딩 되었음을 의미합니다. 
        또한 어휘가 제공되지 않고 레이블이 문자열인 경우 오류가 발생합니다.
    - config	
        런타임 설정을 구성하는 RunConfig 개체입니다.
    - warm_start_from	
        warm_start 할 체크 포인트에 대한 문자열 파일 경로 또는 warm_start를 완전히 구성하기위한 WarmStartSettings 개체입니다. 
        WarmStartSettings 대신 문자열 파일 경로가 제공되면 모든 가중치가 warm_start되고 어휘 및 Tensor 이름이 변경되지 않은 것으로 간주됩니다.
    - loss_reduction	
        중 하나 tf.losses.Reduction를 제외하고 NONE. 배치에 대한 학습 손실을 줄이는 방법을 설명합니다. 기본값은 SUM_OVER_BATCH_SIZE.
    - batch_norm	
        각 은닉층 이후에 일괄 정규화를 사용할지 여부입니다.
    - linear_sparse_combiner	
        범주 형 열이 multivalent(다중 원자가)인 경우 선형 모델을 줄이는 방법을 지정하는 문자열입니다. 
        "mean", "sqrtn"및 "sum" 중 하나 - 이들은 예제 수준의 정규화를 수행하는 효과적으로 다른 방법으로, bag-of-words 기능에 유용 할 수 있습니다.

"""


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
    # => 특성 키:,  ['Velocityms1', 'Cycletimes2', 'LCycletimes3', 'RCycletimes4', 'LStridelengthm5', ... ]
    # "Velocityms1" 배치:  [-1.14285714 -1.02380952 -0.04761905  1.61904762 -0.88095238  0., ... ]
    # 레이블 배치:  [1 0 0 0 1 1 1 0 0 0]

    #  tf.keras.layers.DenseFeatures 층을 사용하여 특정한 특성 열의 결과 확인
    Velocityms0_column = feature_columns[7]
    print(tf.keras.layers.DenseFeatures([Velocityms0_column])(feature_batch).numpy())
    # [[ 0.9661215 ]
    #  [-0.64953274]
    #  [-0.60046726]
    #  [ 1.5665888 ]
    #  [ 0.5186916 ]
    #  [-0.5       ]
    #  [-0.7453271 ]
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

dnn_linear_estimator = tf.estimator.DNNLinearCombinedClassifier(
    n_classes=3,
    # wide(linear) settings
    linear_feature_columns=feature_columns,
    linear_optimizer=tf.keras.optimizers.Ftrl(),
    # deep settings
    dnn_feature_columns=feature_columns,
    dnn_hidden_units=[1024, 512, 256],
    dnn_optimizer=tf.keras.optimizers.Adam(),
    batch_norm=True)

# 모델 학습
dnn_linear_estimator.train(train_input_fn)

# 검증
result = dnn_linear_estimator.evaluate(eval_input_fn)

print('------------------ Eval Predict ------------------')
print(result)

# 테스트 데이터도 스케일링
x_test_scaled = rs_scaler.transform(dftest)

x_test_scaled_df = pd.DataFrame(x_test_scaled, columns=dftest.columns)

# 훈련된 DNN 모델로 테스트 데이터 셋 예측 (단, 테스트 데이터는 label 없이 feature로만 구성))
test_input_fn = make_input_fn_test(x_test_scaled_df, num_epochs=1, shuffle=False)

print('------------------ Test Predict ------------------')
pred = list(dnn_linear_estimator.predict(test_input_fn))
pred_df = pd.DataFrame(pred)

print(pred_df)
print('------------------------------------------')
print(pred_df['class_ids'])

class_ids = pred_df['class_ids'].astype('float64')

print('------------------ Test DNN ------------------------')
print(confusion_matrix(y_test, class_ids))
print(classification_report(y_test, class_ids, target_names=['class 0', 'class 1', 'class 2']))

# ------------------ Eval Predict ------------------
# {'accuracy': 0.92072016, 'average_loss': 0.27124947, 'loss': 0.27123243, 'global_step': 3960}

# ------------------ Test Predict ------------------
#                                    logits  ...         all_classes
# 0      [5.1503773, 2.8849397, -7.6089883]  ...  [b'0', b'1', b'2']
# 1       [1.077478, 1.2605412, -1.6917788]  ...  [b'0', b'1', b'2']
# 2     [-2.4684668, -2.3153973, 6.3002863]  ...  [b'0', b'1', b'2']
# 3     [2.2453156, -0.16748896, -5.857315]  ...  [b'0', b'1', b'2']
# 4       [-1.17754, -3.9122028, 5.8532877]  ...  [b'0', b'1', b'2']
# ...                                   ...  ...                 ...
# 3953  [5.1322002, -2.8321505, -1.5593089]  ...  [b'0', b'1', b'2']
# 3954  [3.2683232, -1.1346982, -6.2442546]  ...  [b'0', b'1', b'2']
# 3955   [0.9269015, -4.870451, -2.9639611]  ...  [b'0', b'1', b'2']
# 3956   [-0.5475174, 5.3813295, -4.427843]  ...  [b'0', b'1', b'2']
# 3957  [-0.24211314, -2.5587118, 4.804173]  ...  [b'0', b'1', b'2']
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
# 3955    [0]
# 3956    [1]
# 3957    [2]

# Name: class_ids, Length: 3958, dtype: object

# ------------------ Test DNN ------------------------
# [[1936   24   28]
#  [ 159 1167   14]
#  [  63   31  536]]
#               precision    recall  f1-score   support
#
#      class 0       0.90      0.97      0.93      1988
#      class 1       0.95      0.87      0.91      1340
#      class 2       0.93      0.85      0.89       630
#
#     accuracy                           0.92      3958
#    macro avg       0.93      0.90      0.91      3958
# weighted avg       0.92      0.92      0.92      3958


# Estimator Model save
serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
    tf.feature_column.make_parse_example_spec(feature_columns))

export_path = dnn_linear_estimator.export_saved_model("new_poma_dnn_linear_model_210518", serving_input_fn)






