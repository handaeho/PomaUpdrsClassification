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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'


def poma_load_dataset():
    poma_3class = pd.read_csv('../../dataset/real_poma_3class_dataset_210518.csv')

    poma_dataset_3class = poma_3class.copy()

    poma_dataset_3class = poma_dataset_3class.sample(n=1000)

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

    updrs_dataset_3class = updrs_dataset_3class_all.sample(n=1000)

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

        interpreter.resize_tensor_input(input_index, [test_data.shape[0], test_data. shape[1]])

        interpreter.allocate_tensors()

        interpreter.set_tensor(input_index, test_data)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest probability.
        output = interpreter.tensor(126)    # index 126 is {'name': 'head/predictions/probabilities'}
        # print(output())
        digit = np.argmax(output()[0])  # highest probability value's index

        prediction_digits.append(digit)

    print('Prediction labels >>', 'total length:', len(prediction_digits), '/', 'example:', prediction_digits[:20])
    print('Real labels >>', 'total length:', len(labels), '/', 'example:', labels[:20])

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

    poma_model_path = '../tensorflow/dnn_linear_combined_classifier/poma_dnn_linear_model.tflite'
    updrs_model_path = '../tensorflow/dnn_linear_combined_classifier/updrs_dnn_linear_model.tflite'

    poma_acc = tf_lite_inference(poma_model_path, poma_x, poma_y)
    print('The POMA accuracy predicted by the DNN Linear TF-Lite Model: ', poma_acc)


    print('---------------------------------------------------------------------------------------------------------')

    updrs_acc = tf_lite_inference(updrs_model_path, updrs_x, updrs_y)
    print('The UPDRS accuracy predicted by the DNN Linear TF-Lite Model: ', updrs_acc)

