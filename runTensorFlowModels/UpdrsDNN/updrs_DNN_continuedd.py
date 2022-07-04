import datetime
import os
import keras.saving.saved_model.model_serialization
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping

# print('tensorflow version:', tf.__version__)
tf.random.set_seed(211116)


def load_dataset_from_elastic_search():
    """
    Load data-set from Elastic search

    :return: DataFrame form ES
    """
    # es = Elasticsearch('[192.168.0.173]:9200')
    # # print(es.info())
    #
    # index_name = 'foot_logger_poma_updrs_3l_labs_data'
    #
    # s = Search(using=es, index=index_name)
    #
    # df = pd.DataFrame([hit.to_dict() for hit in s.scan()])
    #
    # df = df[['poma_danger_3class', 'updrs_danger_3class', 'Velocity(m/s)', 'Cycle_time(s)', 'L_Cycle_time(s)',
    #          'R_Cycle_time(s)', 'L_Stride_length(m)', 'R_Stride_length(m)', 'L_Stride_per_min(Stride/m)',
    #          'R_Stride_per_min(stride/m)', 'L_Foot_vel.(m/s)', 'R_Foot_vel.(m/s)', 'L_step_time(s)', 'R_step_time(s)',
    #          'L_Step_per_min(step/m)', 'R_step_per_min(step/m)', 'L_Stance_time(s)', 'R_Stance_time(s)',
    #          'L_swing_time(s)', 'R_Swing_time(s)', 'DLST_time(s)', 'DLST_Initial_time(s)', 'DLST_Terminal_time(s)',
    #          'L_Total(%)', 'L_In(%)', 'L_out(%)', 'L_front(%)', 'L_back(%)', 'L1(%)', 'L2(%)', 'L3(%)', 'L4(%)',
    #          'L5(%)', 'L6(%)', 'L7(%)', 'L8(%)', 'R_Total(%)', 'R_In(%)', 'R_out(%)', 'R_front(%)', 'R_back(%)',
    #          'R1(%)', 'R2(%)', 'R3(%)', 'R4(%)', 'R5(%)', 'R6(%)', 'R7(%)', 'R8(%)', 'L1 Balance_Time', 'L2', 'L3',
    #          'L4', 'L5', 'L6', 'L7', 'L8', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'L1-1 Sequence',
    #          'L1-2 Sequence', 'L2-1', 'L2-2', 'L3-1', 'L3-2', 'L4-1', 'L4-2', 'L5-1', 'L5-2', 'L6-1', 'L6-2', 'L7-1',
    #          'L7-2', 'L8-1', 'L8-2', 'R1-1', 'R1-2', 'R2-1', 'R2-2', 'R3-1', 'R3-2', 'R4-1', 'R4-2', 'R5-1', 'R5-2',
    #          'R6-1', 'R6-2', 'R7-1', 'R7-2', 'R8-1', 'R8-2']]

    # df = pd.read_csv('../../dataset/real_poma_3class_dataset_210518.csv')
    df = pd.read_csv('../../dataset/real_updrs_3class_dataset_210518.csv')
    df['poma_danger_3class'] = 0

    return df


class MakeUpdrsDnnModelContinued:
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

        # 필요없는 'poma_danger_3class' 컬럼은 삭제
        dataset_3class.drop(['poma_danger_3class'], axis=1, inplace=True)

        # 'updrs_danger_3class' 컬럼 pop -> 라벨
        labels = dataset_3class.pop('updrs_danger_3class').values

        # split dataset to raw train set, eval set.
        features_train, features_test, labels_train, labels_test = train_test_split(dataset_3class, labels,
                                                                                    test_size=0.2, shuffle=True, random_state=2021)

        features_train_x, features_eval, labels_train_x, labels_eval = train_test_split(features_train, labels_train,
                                                                                        test_size=0.2, shuffle=True, random_state=2021)

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

    def model_train_test_save_checkpoint(self, x_train_scaled, x_eval_scaled, x_test_scaled,
                                         y_train_encoding, y_eval_encoding, y_test_encoding, labels_test, saved_model_path):
        """
        Model train and save as H5.

        :param x_train_scaled: scaled train dataset's features
        :param x_eval_scaled: scaled evaluation dataset's features
        :param x_test_scaled: scaled test dataset's features
        :param y_train_encoding: encoded train dataset's labels
        :param y_eval_encoding: encoded evaluation dataset's labels
        :param y_test_encoding: encoded test dataset's labels
        :param labels_test: real test dataset's labels
        :param saved_model_path: saved model name

        :return: model_name, save_path
        """
        epoch = 1000
        batch_size = 64

        if not os.path.exists(saved_model_path):
            os.mkdir(saved_model_path)

        # TODO: 파일 경로는 어디에 저장할건데?
        file_name = saved_model_path + 'updrs_dnn_model_{val_accuracy:.4f}.h5'

        model = self.create_model_baseline_dnn()

        model.summary()

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
                  epochs=epoch, batch_size=batch_size, verbose=1, validation_split=0.2, shuffle=True,
                  callbacks=[early_stopping, model_checkpoint, reduce_lr])

        model.evaluate(x_test_scaled, y_test_encoding)

        # returns an array of probability per classes
        pred = model.predict(x_test_scaled)
        # print(pred)

        # position of max probability
        predictions = []
        for i in pred:
            predictions.append(np.argmax(i))

        # print(y_test)
        # print(predictions)

        print('------------------ Test DNN ------------------------')
        print(confusion_matrix(labels_test, predictions))
        print(classification_report(labels_test, predictions, target_names=['class 0', 'class 1', 'class 2']))

        conf_matrix = confusion_matrix(labels_test, predictions)
        conf_report_dict = classification_report(labels_test, predictions, output_dict=True)

        df_c_matrix = pd.DataFrame(conf_matrix, columns=['class 0', 'class 1', 'class 2'], index=['class 0', 'class 1', 'class 2'])
        df_c_report = pd.DataFrame.from_dict(conf_report_dict).transpose()

        total_accuracy = df_c_report.loc['accuracy'][0]
        total_macro_avg_precision = df_c_report.loc['macro avg'][0]
        total_macro_avg_recall = df_c_report.loc['macro avg'][1]
        total_macro_avg_f1_score = df_c_report.loc['macro avg'][2]
        total_weighted_avg_precision = df_c_report.loc['weighted avg'][0]
        total_weighted_recall = df_c_report.loc['weighted avg'][1]
        total_weighted_f1_score = df_c_report.loc['weighted avg'][2]

        class0_precision = df_c_report['precision'][0]
        class0_recall = df_c_report['recall'][0]
        class0_f1_score = df_c_report['f1-score'][0]
        class0_count = df_c_report['support'][0]

        class1_precision = df_c_report['precision'][1]
        class1_recall = df_c_report['recall'][1]
        class1_f1_score = df_c_report['f1-score'][1]
        class1_count = df_c_report['support'][1]

        class2_precision = df_c_report['precision'][2]
        class2_recall = df_c_report['recall'][2]
        class2_f1_score = df_c_report['f1-score'][2]
        class2_count = df_c_report['support'][2]

        print('--------------------- DataFrame Confusion-Matrix / Classification-Report ---------------------')
        print(df_c_matrix)
        print(df_c_report)

        print('Summary ---------------------')
        print('Accuracy:', total_accuracy,
              'Macro Avg (Precision):', total_macro_avg_precision,
              'Macro Avg (Recall):', total_macro_avg_recall,
              'Macro Avg (F1-score):', total_macro_avg_f1_score,
              'Weighted Avg (Precision):', total_weighted_avg_precision,
              'Weighted Avg (Recall):', total_weighted_recall,
              'Weighted Avg (F1-score):', total_weighted_f1_score
              )

        print('Class 0 ---------------------')
        print('class 0 Precision:', class0_precision, 'class 0 Recall:', class0_recall,
              'class 0 F1-score:', class0_f1_score, 'class 0 Count:', class0_count)

        print('Class 1 ---------------------')
        print('class 1 Precision:', class1_precision, 'class 1 Recall:', class1_recall,
              'class 1 F1-score:', class1_f1_score, 'class 1 Count:', class1_count)

        print('Class 2 ---------------------')
        print('class 2 Precision:', class2_precision, 'class 2 Recall:', class2_recall,
              'class 2 F1-score:', class2_f1_score, 'class 2 Count:', class2_count)

        try:
            # Save the entire model to a HDF5 file.
            # TODO: 파일 저장 경로 수정하기
            h5_model_dir_path = '/home/aiteam/daeho/PomaUpdrs/a_temp_model/'

            model_make_date = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
            model_name = f'updrs_dnn_keras_ver_{model_make_date}.h5'

            save_path = h5_model_dir_path + model_name

            model.save(save_path)

            return model_name, save_path, file_name

        except Exception as e:
            return e

    def model_reload_retrain_checkpoint(self, saved_model_path, x_train_scaled, y_train_encoding, x_test_scaled):
        # TODO: 어디서 불러와서 어떻게 저장해야 될까? + 이전 버전 모델, 새로운 모델 둘 다 가져올수 있는 방법?
        saved_model = keras.models.load_model(saved_model_path)

        # model training metric에서 더 이상 개선이 없다면, training 종료. (patience = n, n번 이상 개선이 없다면 종료)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        # model training 과정 중, 체크 포인트 설정. train 내용을 일정 간격으로 저장하고 추후 다시 load하여 계속 학습 가능.
        model_checkpoint = ModelCheckpoint(saved_model_path, monitor='val_loss', period=100, verbose=1,
                                           save_weights_only=False, save_best_only=True, mode='auto')

        # 모델 개선이 없는 경우, learning rate를 조정해 개선을 유도.
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1,
                                                         mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

        new_model = saved_model.fit(x_train_scaled, y_train_encoding, epochs=5, batch_size=50,
                                    callbacks=[early_stopping, model_checkpoint, reduce_lr])

        # assert_allclose: 두 개체가 원하는 허용 오차까지 같지 않으면 AssertionError를 발생.
        np.testing.assert_allclose(saved_model.predict(x_test_scaled), new_model.predict(x_test_scaled), 1e-5)

    ##############################################################################################################


if __name__ == '__main__':
    r = MakeUpdrsDnnModelContinued()

    df = load_dataset_from_elastic_search()

    saved_model_path = '/home/aiteam/daeho/PomaUpdrs/checkpoint_models/DNN_checkpoint_model/updrs_DNN_checkpoint_Model/'

    x_train, x_eval, x_test, y_train, y_eval, y_test, labels_test = r.load_dataset(df)

    print(len(x_train), len(x_eval), len(x_test))  # 25404 6352 7940
    print(len(y_train), len(y_eval), len(y_test), len(labels_test))  # 25404 6352 7940 7940

    model_name, save_path, file_name = r.model_train_test_save_checkpoint(x_train_scaled=x_train, x_eval_scaled=x_eval, x_test_scaled=x_test,
                                                                          y_train_encoding=y_train, y_eval_encoding=y_eval, y_test_encoding=y_test,
                                                                          labels_test=labels_test, saved_model_path=saved_model_path)

    print(model_name, save_path, file_name)

    # r.model_reload_retrain_checkpoint(saved_model_path=saved_model_path, x_train_scaled=x_train, y_train_encoding=y_train, x_test_scaled=x_test)

    # --------------------- DataFrame Confusion-Matrix / Classification-Report ---------------------
    #          class 0  class 1  class 2
    # class 0     1745       62        8
    # class 1      147     1353       36
    # class 2       42       55      510
    #               precision    recall  f1-score      support
    # 0              0.902275  0.961433  0.930915  1815.000000
    # 1              0.920408  0.880859  0.900200  1536.000000
    # 2              0.920578  0.840198  0.878553   607.000000
    # accuracy       0.911572  0.911572  0.911572     0.911572
    # macro avg      0.914420  0.894163  0.903222  3958.000000
    # weighted avg   0.912119  0.911572  0.910965  3958.000000