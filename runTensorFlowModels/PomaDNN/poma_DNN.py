import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# print('tensorflow version:', tf.__version__)
tf.random.set_seed(210813)


class MakePomaDnnModel:
    def __init__(self):
        print('Start Train and Save DNN Model with POMA DataSet')

    def load_dataset(self, dataset_from_es):
        """
        Method to train test split and scale data loaded from Elastic-Search.

        :param dataset_from_es: DataSet from ES (DataFrame)

        :return: Scaled train set, eval set / encoded train set labels, eval set labels
        """
        dataset_3class = dataset_from_es.copy()

        # Drop rows containing nan values from dataset
        dataset_3class.dropna(axis=0, inplace=True)

        # 필요없는 'updrs_danger_3class' 컬럼은 삭제
        dataset_3class.drop(['updrs_danger_3class'], axis=1, inplace=True)

        # 'poma_danger_3class' 컬럼 pop -> 라벨
        labels = dataset_3class.pop('poma_danger_3class').values

        # split dataset to raw train set, eval set.
        features_train, features_test, labels_train, labels_test = train_test_split(dataset_3class, labels,
                                                                                    test_size=0.2, shuffle=True,
                                                                                    random_state=1220)

        features_train_x, features_eval, labels_train_x, labels_eval = train_test_split(features_train, labels_train,
                                                                                        test_size=0.2, shuffle=True, random_state=1220)

        # 변형 객체 생성
        rs_scaler = RobustScaler()

        # 훈련 데이터 스케일링
        x_train_scaled = rs_scaler.fit_transform(features_train_x)

        # 검증 데이터의 스케일링
        x_eval_scaled = rs_scaler.transform(features_eval)

        # 테스트 데이터의 스케일링
        x_test_scaled = rs_scaler.transform(features_test)

        # label One-hot encoding
        y_train_encoding = to_categorical(labels_train_x, 3)
        y_eval_encoding = to_categorical(labels_eval, 3)
        y_test_encoding = to_categorical(labels_test, 3)

        return x_train_scaled, x_eval_scaled, x_test_scaled, y_train_encoding, y_eval_encoding, y_test_encoding, labels_test

    ##############################################################################################################

    # dnn model
    def create_model_baseline_dnn(self):
        """
        Create DNN Model with 3 dense layers.

        :return: DNN Model
        """
        # create model
        model = Sequential()

        model.add(Dense(units=1024, activation='relu', input_shape=(95,)))

        model.add(Dense(units=512, activation='relu'))

        model.add(Dense(units=256, activation='relu'))

        model.add(Dense(3, activation='softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])

        return model

    ##############################################################################################################

    def model_train_test_save(self, x_train_scaled, x_eval_scaled, x_test_scaled, y_train_encoding, y_eval_encoding, y_test_encoding, labels_test):
        """
        Model train and save as H5.

        :param x_train_scaled: scaled train dataset's features
        :param x_eval_scaled: scaled evaluation dataset's features
        :param x_test_scaled: scaled test dataset's features
        :param y_train_encoding: encoded train dataset's labels
        :param y_eval_encoding: encoded evaluation dataset's labels
        :param y_test_encoding: encoded test dataset's labels
        :param labels_test: real test dataset's labels

        :return: model_name, save_path
        """
        model = self.create_model_baseline_dnn()

        model.summary()

        date_time = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

        saved_check_point_model_path = '/home/aiteam/daeho/PomaUpdrs/checkpoint_models/DNN_checkpoint_model/poma_DNN_checkpoint_Model/'

        file_name = saved_check_point_model_path + f'poma_dnn_check_point_model_{date_time}.h5'

        # model training metric에서 더 이상 개선이 없다면, training 종료. (patience = n, n번 이상 개선이 없다면 종료)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        # model training 과정 중, 체크 포인트 설정. train 내용을 일정 간격으로 저장하고 추후 다시 load하여 계속 학습 가능.
        model_checkpoint = ModelCheckpoint(file_name, monitor='val_loss', period=100, verbose=1,
                                           save_weights_only=False, save_best_only=True, mode='auto')

        # 모델 개선이 없는 경우, learning rate를 조정해 개선을 유도.
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1,
                                                         mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

        model.fit(x_train_scaled, y_train_encoding,
                  validation_data=(x_eval_scaled, y_eval_encoding),
                  epochs=1000, verbose=0, validation_split=0.2,
                  callbacks=[early_stopping, model_checkpoint, reduce_lr])

        model.evaluate(x_test_scaled, y_test_encoding)

        # returns an array of probability per classes
        pred = model.predict(x_test_scaled, batch_size=128)
        # print(pred)

        # position of max probability
        predictions = []
        for i in pred:
            predictions.append(np.argmax(i))

        # print(y_test)
        # print(predictions)

        print('------------------ Test DNN Confusion-Matrix / Classification-Report ------------------------')
        print(confusion_matrix(labels_test, predictions))
        print(classification_report(labels_test, predictions, target_names=['class 0', 'class 1', 'class 2']))

        try:
            # Save the entire model to a HDF5 file.
            h5_model_dir_path = '/home/aiteam/daeho/PomaUpdrs/H5_Pb_models/DNN_H5_models/poma_DNN_H5_Model/'

            model_make_date = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
            model_name = f'poma_dnn_keras_ver_{model_make_date}.h5'

            save_path = h5_model_dir_path + model_name

            model.save(save_path)

            return model_name, save_path

        except Exception as e:
            return e

    ##############################################################################################################
