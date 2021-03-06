import datetime

import numpy as np
import tensorflow as tf

from tensorflow.python.keras import Sequential, models
from tensorflow.python.keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# print('tensorflow version:', tf.__version__)
tf.random.set_seed(210813)


# dnn model
def create_model_baseline_dnn():
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


class MakePomaDnnModel:
    def __init__(self, dataset):
        print('Start Train and Save DNN Model with POMA DataSet - 2022.05.31 Ver.')
        self.dataset = dataset

        self.date_time = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

        # Save model with check-point
        self.check_point_model_path = '/home/aiteam/daeho/PomaUpdrs/checkpoint_models/DNN_checkpoint_model/poma_DNN_checkpoint_Model/'

        # Save the entire model to a HDF5 file.
        self.h5_model_path = '/home/aiteam/daeho/PomaUpdrs/H5_Pb_models/DNN_H5_models/poma_DNN_H5_Model/'

    def load_dataset(self):
        """
        Method to train test split and scale stream data loaded from 3L-labs.

        This dataset is for continual train and evaluation.

        Additional training is performed on the existing model with a new dataset.

        :return: Scaled train set, eval set / encoded train set labels, eval set labels
        """
        dataset_3class = self.dataset.copy()

        # Drop rows containing nan values from dataset
        dataset_3class.dropna(axis=0, inplace=True)

        # ???????????? 'updrs_danger_3class' ????????? ??????
        dataset_3class.drop(['updrs_danger_3class'], axis=1, inplace=True)

        # 'poma_danger_3class' ?????? pop -> ??????
        labels = dataset_3class.pop('poma_danger_3class').values

        # split dataset to raw train set, eval set.
        features_train, features_test, labels_train, labels_test = train_test_split(dataset_3class, labels,
                                                                                    test_size=0.2, shuffle=True,
                                                                                    random_state=1220)

        features_train_x, features_eval, labels_train_x, labels_eval = train_test_split(features_train, labels_train,
                                                                                        test_size=0.2, shuffle=True, random_state=1220)

        # ?????? ?????? ??????
        rs_scaler = RobustScaler()

        # ?????? ????????? ????????????
        x_train_scaled = rs_scaler.fit_transform(features_train_x)

        # ?????? ???????????? ????????????
        x_eval_scaled = rs_scaler.transform(features_eval)

        # ????????? ???????????? ????????????
        x_test_scaled = rs_scaler.transform(features_test)

        # label One-hot encoding
        y_train_encoding = to_categorical(labels_train_x, 3)
        y_eval_encoding = to_categorical(labels_eval, 3)
        y_test_encoding = to_categorical(labels_test, 3)

        return x_train_scaled, x_eval_scaled, x_test_scaled, y_train_encoding, y_eval_encoding, y_test_encoding, labels_test

    def input_data(self):
        """
        scaling to receive data. not included label

        :return: scaled input data
        """
        dataset_3class = self.dataset.copy()

        # dataset_3class.drop(['poma_danger_3class', 'updrs_danger_3class'], axis=1, inplace=True)

        # ?????? ?????? ??????
        rs_scaler = RobustScaler()

        # input data ????????????
        input_x_scaled = rs_scaler.fit_transform(dataset_3class)

        return input_x_scaled

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
        model = create_model_baseline_dnn()

        model.summary()

        # model training metric?????? ??? ?????? ????????? ?????????, training ??????. (patience = n, n??? ?????? ????????? ????????? ??????)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        # model training ?????? ???, ?????? ????????? ??????. train ????????? ?????? ???????????? ???????????? ?????? ?????? load?????? ?????? ?????? ??????.
        check_point_model_file_name = self.check_point_model_path + f'poma_dnn_check_point_model_{self.date_time}.h5'

        model_checkpoint = ModelCheckpoint(check_point_model_file_name, monitor='val_loss', period=100, verbose=1, save_weights_only=False, save_best_only=True, mode='auto')

        # ?????? ????????? ?????? ??????, learning rate??? ????????? ????????? ??????.
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

        model.fit(x_train_scaled, y_train_encoding,
                  validation_data=(x_eval_scaled, y_eval_encoding), epochs=1000, verbose=0, validation_split=0.2, callbacks=[early_stopping, model_checkpoint, reduce_lr])

        model.evaluate(x_test_scaled, y_test_encoding)

        # returns an array of probability per classes
        pred = model.predict(x_test_scaled, batch_size=128)
        # print(pred)

        # position of max probability
        predictions = []
        for i in pred:
            predictions.append(np.argmax(i))

        # print(labels_test)  # true labels
        # print(predictions)  # prediction labels

        print('------------------ Test DNN Confusion-Matrix / Classification-Report ------------------------')
        print(confusion_matrix(labels_test, predictions))
        print(classification_report(labels_test, predictions, target_names=['class 0', 'class 1', 'class 2']))

        # F1-Score
        f1 = round(f1_score(labels_test, predictions, average='micro'), 4)
        print(f1)

        try:
            model_name = f'poma_dnn_keras_ver_{self.date_time}_{f1}.h5'

            save_path = self.h5_model_path + model_name

            model.save(save_path)

            print(model_name)

        except Exception as e:
            return e

        return labels_test, predictions

    def saved_model_predict(self, model_name, input_data):
        """
        load to saved model and predict using input data.

        :param model_name: saved model name
        :param input_data: receive data

        :return: result of prediction
        """
        saved_model_path = self.h5_model_path + model_name

        saved_model = models.load_model(saved_model_path)

        predicted_result = saved_model.predict(input_data)

        return predicted_result




