# -*- coding:utf-8 -*-
"""
It checks the file list in the file receiving folder (DataSet_From_3L_labs).
And if there is a change, executes 'run.py' to execute the model creation automation logic.
"""
import datetime
import os
import pickle
import tensorflow as tf
import warnings

from run import Run
from run import load_dataset_from_elastic_search

# print(os.path.realpath(__file__))  # 현재 파일 실제 경로
# print(os.path.abspath(__file__))  # 현재 파일 절대 경로
print('tensorflow version:', tf.__version__)

warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

data_file_path = "/home/aiteam/daeho/PomaUpdrs/DataSet_From_3L_labs"
data_file_list_path = data_file_path + '/data_file_list.txt'
current_file_list = os.listdir(data_file_path)


def startRun():
    with open(data_file_list_path, 'rb') as f:
        data = pickle.load(f)

    print("Now file list: ", data)

    if set(data) != set(current_file_list):
        run = Run()

        x = load_dataset_from_elastic_search()
        print(x)

        print('-------------------- DNN Model start --------------------')
        poma_dnn_acc, poma_dnn_tflite_save_name, poma_dnn_tflite_save_path = run.run_poma_dnn()
        print('@@@@@@@@@@@@@@@ The POMA accuracy predicted by the DNN TF-Lite Model:', poma_dnn_acc)
        print('@@@@@@@@@@@@@@@ Model Name =>', poma_dnn_tflite_save_name)
        print('@@@@@@@@@@@@@@@ Model Path =>', poma_dnn_tflite_save_path)

        #

        print('----------------------------------------------------------------------------------------------------')

        updrs_dnn_acc, updrs_dnn_tflite_save_name, updrs_dnn_tflite_save_path = run.run_updrs_dnn()
        print('@@@@@@@@@@@@@@@ The UPDRS accuracy predicted by the DNN TF-Lite Model:', updrs_dnn_acc)
        print('@@@@@@@@@@@@@@@ Model Name =>', updrs_dnn_tflite_save_name)
        print('@@@@@@@@@@@@@@@ Model Path =>', updrs_dnn_tflite_save_path)

        #

        print('-------------------- DNN-Linear combined Model start --------------------')
        poma_dnn_linear_acc, poma_dnn_linear_tflite_save_name, poma_dnn_linear_tflite_save_path = run.run_poma_dnn_linear()
        print('@@@@@@@@@@@@@@@ The POMA accuracy predicted by the DNN-Linear combined TF-Lite Model:', poma_dnn_linear_acc)
        print('@@@@@@@@@@@@@@@ Model Name =>', poma_dnn_linear_tflite_save_name)
        print('@@@@@@@@@@@@@@@ Model Path =>', poma_dnn_linear_tflite_save_path)

        #

        print('----------------------------------------------------------------------------------------------------')

        updrs_dnn_linear_acc, updrs_dnn_linear_tflite_save_name, updrs_dnn_linear_tflite_save_path = run.run_updrs_dnn_linear()
        print('@@@@@@@@@@@@@@@ The UPDRS accuracy predicted by the DNN-Linear combined TF-Lite Model:', updrs_dnn_linear_acc)
        print('@@@@@@@@@@@@@@@ Model Name =>', updrs_dnn_linear_tflite_save_name)
        print('@@@@@@@@@@@@@@@ Model Path =>', updrs_dnn_linear_tflite_save_path)

        #

        print('-------------------- Gradient Boosting Trees Model start --------------------')
        poma_gradient_boosting_tflite_save_name, poma_gradient_boosting_tflite_save_path = run.run_poma_gradient_boosting()
        print('@@@@@@@@@@@@@@@ Model Name =>', poma_gradient_boosting_tflite_save_name)
        print('@@@@@@@@@@@@@@@ Model Path =>', poma_gradient_boosting_tflite_save_path)

        #

        print('----------------------------------------------------------------------------------------------------')

        updrs_gradient_boosting_tflite_save_name, updrs_gradient_boosting_tflite_save_path = run.run_updrs_gradient_boosting()
        print('@@@@@@@@@@@@@@@ Model Name =>', updrs_gradient_boosting_tflite_save_name)
        print('@@@@@@@@@@@@@@@ Model Path =>', updrs_gradient_boosting_tflite_save_path)

        running_date = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        with open(data_file_list_path, 'wb') as f:
            pickle.dump(current_file_list, f)

        print(current_file_list)

        return f'---------------------------------------- Run Completed! at {running_date} ----------------------------------------'

    else:
        return '---------------------------------------- Nothing Happened... ----------------------------------------'


if __name__ == '__main__':
    start_run = startRun()

    print(start_run)
