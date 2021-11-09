import tensorflow as tf
import numpy as np
import os
import time

from sklearn.preprocessing import RobustScaler

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'


class UpdrsDNNInferenceTfLiteModel:
    def __init__(self):
        print('Inference Test for DNN TF-Lite Model with UPDRS DataSet')

    def updrs_load_dataset(self, test_dataset):
        dataset_3class = test_dataset.copy()

        # Drop rows containing nan values from dataset
        dataset_3class.dropna(axis=0, inplace=True)

        # 필요없는 'poma_danger_3class' 컬럼은 삭제
        dataset_3class.drop(['poma_danger_3class'], axis=1, inplace=True)

        dataset_3class = dataset_3class.sample(n=10000)

        # 'danger' 컬럼 pop -> 라벨
        labels = dataset_3class.pop('updrs_danger_3class').values

        features = dataset_3class.loc[:, dataset_3class.columns != 'updrs_danger_3class'].values

        # 변형 객체 생성
        rs_scaler = RobustScaler().fit(features)

        # 데이터 스케일링
        features_scaled = rs_scaler.transform(features)

        return features_scaled, labels

    # A helper function to evaluate the TF Lite model using "test" dataset.
    def tf_lite_inference(self, model_path, inputs, labels):
        start = time.time()

        # TF-Lite 모델 파일 로드
        interpreter = tf.lite.Interpreter(model_path=model_path)

        interpreter.allocate_tensors()

        input_index = interpreter.get_input_details()[0]["index"]

        output_index = interpreter.get_output_details()[0]["index"]

        prediction_digits = []

        for data in inputs:
            # Pre-processing: add batch dimension and convert to float32 to match with the model's input data format.
            test_data = np.expand_dims(data, axis=0).astype(np.float32)

            interpreter.resize_tensor_input(input_index, [test_data.shape[0], test_data.shape[1]])

            interpreter.allocate_tensors()

            interpreter.set_tensor(input_index, test_data)

            # Run inference.
            interpreter.invoke()

            # Post-processing: remove batch dimension and find the digit with highest probability.
            output = interpreter.get_tensor(output_index)

            # print(output())
            digit = np.argmax(output[0])  # highest probability value's index

            prediction_digits.append(digit)

        # Compare prediction results with ground truth labels to calculate accuracy.
        accurate_count = 0

        for index in range(len(prediction_digits)):
            if prediction_digits[index] == labels[index]:
                accurate_count += 1

        accuracy = accurate_count * 1.0 / len(prediction_digits)

        end = time.time()
        print('@@@@@@@@@@@@@@@ Model running time elapsed:', end - start)

        return accuracy
