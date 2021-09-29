from sklearn.preprocessing import RobustScaler

import tensorflow as tf
import numpy as np
import pandas as pd
import re
import os
import time

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def poma_load_dataset():
    poma_3class = pd.read_csv('../../dataset/real_poma_3class_dataset_210518.csv')

    poma_dataset_3class = poma_3class.copy()

    poma_dataset_3class = poma_dataset_3class.sample(n=100)

    # TF2 Pipe-line에는 컬럼명에 특수문자 불가.
    cols = [re.sub(r'[\W_]', "", i) for i in poma_dataset_3class.columns]

    for i in range(len(cols)):
        cols[i] = cols[i] + str(i)

    poma_dataset_3class.columns = cols

    print(poma_dataset_3class)

    # 'danger' 컬럼 -> 라벨
    features = poma_dataset_3class.iloc[:, 1:].values
    labels = poma_dataset_3class['pomadanger3class0'].values

    # 변형 객체 생성 / 모수 분포 저장 (fit)
    rs_scaler = RobustScaler().fit(features)

    # 데이터 스케일링 (transform)
    features_scaled = rs_scaler.transform(features)

    return features_scaled, labels


def updrs_load_dataset():
    updrs_3class = pd.read_csv('../../dataset/real_updrs_3class_dataset_210518.csv')

    updrs_dataset_3class_all = updrs_3class.copy()

    updrs_dataset_3class = updrs_dataset_3class_all.sample(n=100)

    # TF2 Pipe-line에는 컬럼명에 특수문자 불가.
    cols = [re.sub(r'[\W_]', "", i) for i in updrs_dataset_3class.columns]

    for i in range(len(cols)):
        cols[i] = cols[i] + str(i)

    updrs_dataset_3class.columns = cols

    print(updrs_dataset_3class)

    # 'danger' 컬럼 -> 라벨
    features = updrs_dataset_3class.iloc[:, 1:].values
    labels = updrs_dataset_3class['updrsdanger3class0'].values

    # 변형 객체 생성
    rs_scaler = RobustScaler().fit(features)

    # 데이터 스케일링
    features_scaled = rs_scaler.transform(features)

    return features_scaled, labels


# A helper function to evaluate the TF Lite model using "test" dataset.
def tf_lite_inference(model_path, inputs, labels):
    start = time.time()

    # TF-Lite 모델 파일 로드
    interpreter = tf.lite.Interpreter(model_path=model_path)

    interpreter.allocate_tensors()

    # for item in interpreter.get_tensor_details():
    #     for key in item.keys():
    #         print("%s : %s" % (key, item[key]))
    #     print("")

    print(interpreter.get_input_details())
    print(interpreter.get_output_details())

    input_index = interpreter.get_input_details()[0]["index"]
    # output_index = interpreter.get_output_details()[0]["index"]

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []

    for data in inputs:
        # Pre-processing: add batch dimension and convert to float32 to match with the model's input data format.
        test_data = np.expand_dims(data, axis=0).astype(np.float32)
        # print(data)
        # print(np.shape(data))   # (95,)
        # print(test_data)
        # print(np.shape(test_data))  # (1, 95)

        interpreter.resize_tensor_input(input_index, [test_data.shape[0], test_data.shape[1]])

        interpreter.allocate_tensors()

        interpreter.set_tensor(input_index, test_data)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest probability.
        output = interpreter.tensor(115)    # index 115 is {'name': 'head/predictions/probabilities'}
        # print(output())
        digit = np.argmax(output()[0])

        prediction_digits.append(digit)

    print('Prediction labels :', len(prediction_digits), prediction_digits)
    print('Real labels :', len(labels), labels)

    # Compare prediction results with ground truth labels to calculate accuracy.
    accurate_count = 0

    for index in range(len(prediction_digits)):
        if prediction_digits[index] == labels[index]:
            accurate_count += 1

    accuracy = accurate_count * 1.0 / len(prediction_digits)

    end = time.time()
    print('Model running time elapsed:', end - start)

    return accuracy


if __name__ == '__main__':
    poma_x, poma_y = poma_load_dataset()
    # print(poma_x)
    # print(poma_y)

    updrs_x, updrs_y = updrs_load_dataset()
    # print(updrs_x)
    # print(updrs_y)

    poma_model_path = '../tensorflow/gradient_boosting/new_poma_gradient_boosting_model_210610_quantization_0001.tflite'
    updrs_model_path = '../tensorflow/gradient_boosting/new_updrs_gradient_boosting_model_210610_quantization_0001.tflite'

    poma_acc = tf_lite_inference(poma_model_path, poma_x, poma_y)
    print('POMA TF-Lite Model acc: ', poma_acc)

    print('---------------------------------------------------------------------------------------------------------')

    updrs_acc = tf_lite_inference(updrs_model_path, updrs_x, updrs_y)
    print('UPDRS TF-Lite Model acc: ', updrs_acc)

    """
        RuntimeError: Encountered unresolved custom op: BoostedTreesEnsembleResourceHandleOp.Node number 0 
                     (BoostedTreesEnsembleResourceHandleOp) failed to prepare.
        => Boosting Tree 의 경우, TF-Lite 변환할 때 'converter.allow_custom_ops = True' 설정 해야 하는데 그러면 inference 안되는데?

        >> 깃허브의 텐서플로우 개발팀 답변. (https://github.com/tensorflow/tensorflow/issues/34350)
            => Unfortunately, models converted with TensorFlow select ops cannot be run in Python interpreters. 
                (불행히도 TensorFlow 선택 작업으로 변환 된 모델은 Python 인터프리터에서 실행할 수 없습니다.)
        
        그럼 일단 TF-Lite 모델로 변환은 되었으니 안드로이드에서 테스트 해봐야 한다?
    """
