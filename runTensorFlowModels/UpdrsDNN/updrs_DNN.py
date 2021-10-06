import datetime
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

print('tensorflow version:', tf.__version__)
tf.random.set_seed(210813)


class MakeUpdrsDnnModel:
    def __init__(self):
        print('Start Train and Save DNN Model with UPDRS DataSet')

    def load_dataset(self, dataset_from_es):
        """
        Method to train test split and scale data loaded from Elastic-Search.

        :param dataset_from_es: DataSet from ES (DataFrame)

        :return: Scaled train set, eval set / encoded train labels, eval labels
        """
        dataset_3class = dataset_from_es.copy()

        # Drop rows containing nan values from dataset
        dataset_3class.dropna(axis=0, inplace=True)

        # 필요없는 'poma_danger_3class' 컬럼은 삭제
        dataset_3class.drop(['poma_danger_3class'], axis=1, inplace=True)

        # 'danger' 컬럼 pop -> 라벨
        labels = dataset_3class.pop('updrs_danger_3class').values

        # split dataset to raw train set, eval set.
        features_train, features_test, labels_train, labels_test = train_test_split(dataset_3class, labels,
                                                                                    test_size=0.2, shuffle=True,
                                                                                    random_state=1220)

        features_train_x, features_eval, labels_train_x, labels_eval = train_test_split(features_train, labels_train,
                                                                                        test_size=0.2, shuffle=True,
                                                                                        random_state=1220)
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

        return x_train_scaled, x_eval_scaled, x_test_scaled, y_train_encoding, y_eval_encoding, y_test_encoding

    ##############################################################################################################

    # baseline model
    def create_model_baseline(self):
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

    def model_train_test_save(self, x_train_scaled, x_eval_scaled, x_test_scaled, y_train_encoding, y_eval_encoding, y_test_encoding):
        """
        Model train and save as H5.

        :param x_train_scaled: scaled train dataset's features
        :param x_eval_scaled: scaled evaluation dataset's features
        :param x_test_scaled: scaled test dataset's features
        :param y_train_encoding: encoded train dataset's labels
        :param y_eval_encoding: encoded evaluation dataset's labels
        :param y_test_encoding: encoded test dataset's labels

        :return: model_name, save_path
        """
        model = self.create_model_baseline()

        model.summary()

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

        model.fit(x_train_scaled, y_train_encoding,
                  validation_data=[[x_eval_scaled, y_eval_encoding]],
                  epochs=1000, verbose=1, validation_split=0.2, callbacks=[callback])

        model.evaluate(x_test_scaled, y_test_encoding, batch_size=64)

        try:
            # Save the entire model to a HDF5 file.
            h5_model_dir_path = '/home/aiteam/daeho/PomaUpdrs/H5_Pb_models/DNN_H5_models/updrs_DNN_H5_Model/'

            model_make_date = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
            model_name = f'updrs_dnn_keras_ver_{model_make_date}.h5'

            save_path = h5_model_dir_path + model_name

            model.save(save_path)

            return model_name, save_path

        except Exception as e:
            return e

    ##############################################################################################################
