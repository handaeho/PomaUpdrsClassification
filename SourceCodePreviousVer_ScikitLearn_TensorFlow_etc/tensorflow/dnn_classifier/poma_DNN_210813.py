import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import re

from shutil import copytree
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from os import path, listdir

print('tensorflow version:', tf.__version__)
tf.random.set_seed(210813)


def load_dataset():
    """
    Method to train test split and scale data loaded from Elastic-Search.

    :return: Scaled X, Y features, labels.
    """
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

    # 'danger' 컬럼 pop -> 라벨
    labels = poma_dataset_3class.pop('pomadanger3class0').values

    # split dataset to raw train set, test set.
    features_x, features_test, labels_y, labels_test = train_test_split(poma_dataset_3class, labels,
                                                                        test_size=0.2, shuffle=True, random_state=1220)

    # split raw train set to real train set, eval set.
    features_train, features_eval, labels_train, labels_eval = train_test_split(features_x, labels_y,
                                                                                test_size=0.2, shuffle=True,
                                                                                random_state=1220)
    # print(x_train.values, x_train.shape)
    # print(x_test.values, x_test.shape)
    # print(y_train, y_train.shape)
    # print(y_test, y_test.shape)

    # 변형 객체 생성
    rs_scaler = RobustScaler()

    # 훈련 데이터 스케일링
    x_train_scaled = rs_scaler.fit_transform(features_train)

    # 검증 데이터의 스케일링
    x_eval_scaled = rs_scaler.transform(features_eval)

    # 테스트 데이터의 스케일링
    x_test_scaled = rs_scaler.transform(features_test)

    # label One-hot encoding
    y_train_encoding = to_categorical(labels_train, 3)
    y_eval_encoding = to_categorical(labels_eval, 3)

    # print(x_train_scaled, x_train_scaled.shape)
    # print(x_eval_scaled, x_eval_scaled.shape)
    # print(labels_train, labels_train.shape)
    # print(labels_eval, labels_eval.shape)
    # print(y_train_encoding, y_train_encoding.shape)
    # print(y_eval_encoding, y_eval_encoding.shape)

    return x_train_scaled, x_test_scaled, x_eval_scaled, y_train_encoding, y_eval_encoding


##############################################################################################################
class PomaDnnModel:
    # baseline model
    def create_model_baseline(self):
        """
        Create a DNN model.

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

    def model_train_save(self):
        """
        Train DNN model and save as H5.

        :return: DNN model stored as H5
        """
        model = self.create_model_baseline()

        x_train_scaled, x_test_scaled, x_eval_scaled, y_train_encoding, y_eval_encoding = load_dataset()

        model.summary()

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

        model.fit(x_train_scaled, y_train_encoding, epochs=1000, verbose=1, validation_split=0.2, callbacks=[callback])

        model.evaluate(x_eval_scaled, y_eval_encoding)

        # returns an array of probability per classes
        pred = model.predict(x_test_scaled, batch_size=128)
        # print(pred)

        # position of max probability
        predictions = []
        for i in pred:
            predictions.append(np.argmax(i))

        # print(y_test)
        # print(predictions)

        # print('------------------ Test DNN ------------------------')
        # print(confusion_matrix(labels_test, predictions))
        # print(classification_report(labels_test, predictions, target_names=['class 0', 'class 1', 'class 2']))

        # Save the entire model to a HDF5 file.
        h5_model_dir_path = '/home/aiteam/daeho/PomaUpdrs/DNN_H5_models/'
        save_path = h5_model_dir_path + 'poma_dnn_keras_ver' + '{date}.h5'
        save_path.format(date=datetime.datetime)
        model.save(save_path)

        return 0

    ##############################################################################################################
