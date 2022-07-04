import datetime
import os

import keras.saving.saved_model.model_serialization
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, accuracy_score, f1_score, r2_score
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
    es = Elasticsearch('[192.168.0.173]:9200')
    # print(es.info())

    index_name = 'foot_logger_poma_updrs_3l_labs_data'

    s = Search(using=es, index=index_name)

    df = pd.DataFrame([hit.to_dict() for hit in s.scan()])

    df = df[['poma_danger_3class', 'updrs_danger_3class', 'Velocity(m/s)', 'Cycle_time(s)', 'L_Cycle_time(s)',
             'R_Cycle_time(s)', 'L_Stride_length(m)', 'R_Stride_length(m)', 'L_Stride_per_min(Stride/m)',
             'R_Stride_per_min(stride/m)', 'L_Foot_vel.(m/s)', 'R_Foot_vel.(m/s)', 'L_step_time(s)', 'R_step_time(s)',
             'L_Step_per_min(step/m)', 'R_step_per_min(step/m)', 'L_Stance_time(s)', 'R_Stance_time(s)',
             'L_swing_time(s)', 'R_Swing_time(s)', 'DLST_time(s)', 'DLST_Initial_time(s)', 'DLST_Terminal_time(s)',
             'L_Total(%)', 'L_In(%)', 'L_out(%)', 'L_front(%)', 'L_back(%)', 'L1(%)', 'L2(%)', 'L3(%)', 'L4(%)',
             'L5(%)', 'L6(%)', 'L7(%)', 'L8(%)', 'R_Total(%)', 'R_In(%)', 'R_out(%)', 'R_front(%)', 'R_back(%)',
             'R1(%)', 'R2(%)', 'R3(%)', 'R4(%)', 'R5(%)', 'R6(%)', 'R7(%)', 'R8(%)', 'L1 Balance_Time', 'L2', 'L3',
             'L4', 'L5', 'L6', 'L7', 'L8', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'L1-1 Sequence',
             'L1-2 Sequence', 'L2-1', 'L2-2', 'L3-1', 'L3-2', 'L4-1', 'L4-2', 'L5-1', 'L5-2', 'L6-1', 'L6-2', 'L7-1',
             'L7-2', 'L8-1', 'L8-2', 'R1-1', 'R1-2', 'R2-1', 'R2-2', 'R3-1', 'R3-2', 'R4-1', 'R4-2', 'R5-1', 'R5-2',
             'R6-1', 'R6-2', 'R7-1', 'R7-2', 'R8-1', 'R8-2']]

    # df = pd.read_csv('../../dataset/real_poma_3class_dataset_210518.csv')
    # df['updrs_danger_3class'] = 0

    return df


class MakePomaDnnModelContinued:
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

    def model_reload_retrain_checkpoint(self, saved_model_path, x_train_scaled, y_train_encoding, x_test_scaled):
        # TODO: 어디서 불러와서 어떻게 저장해야 될까? + 이전 버전 모델, 새로운 모델 둘 다 가져올수 있는 방법?
        saved_model = tf.keras.models.load_model(saved_model_path)

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
    r = MakePomaDnnModelContinued()

    df = load_dataset_from_elastic_search()

    saved_model_path = '/home/aiteam/daeho/PomaUpdrs/checkpoint_models/DNN_checkpoint_model/poma_DNN_checkpoint_Model/'

    x_train, x_eval, x_test, y_train, y_eval, y_test, labels_test = r.load_dataset(df)

    print(len(x_train), len(x_eval), len(x_test))  # CSV: 25404 6352 7940 / ES: 25411 6353 7942
    print(len(y_train), len(y_eval), len(y_test), len(labels_test))  # CSV: 25404 6352 7940 7940 / ES: 25411 6353 7942 7942











