"""
This is the code to load data, create DNN model, save H5 model, and create TF-Lite file.
"""
import pandas as pd
import sys
import os

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search

# import top-level parent and parent folder paths (append to system path) -> /home/aiteam/daeho/PomaUpdrs
sys.path.append((os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))))
# print(sys.path)

# DNN Model
from runTensorFlowModels.PomaDNN.poma_DNN import MakePomaDnnModel
from runTensorFlowModels.UpdrsDNN.updrs_DNN import MakeUpdrsDnnModel
from runTensorFlowModels.PomaDNN.poma_DNN_convert_h5_to_tflite import PomaDNNConvertH5toTflite
from runTensorFlowModels.UpdrsDNN.updrs_DNN_convert_h5_to_tflite import UpdrsDNNConvertH5toTflite
from runTensorFlowModels.PomaDNN.poma_DNN_keras_ver_tflite_inference import PomaDNNInferenceTfLiteModel
from runTensorFlowModels.UpdrsDNN.updrs_DNN_keras_ver_tflite_inference import UpdrsDNNInferenceTfLiteModel

# DNN Linear combined Model
from runTensorFlowModels.PomaDNNLinear.poma_DNN_Linear import MakePomaDnnLinearModel
from runTensorFlowModels.UpdrsDNNLinear.updrs_DNN_Linear import MakeUpdrsDnnLinearModel
from runTensorFlowModels.PomaDNNLinear.poma_DNN_Linear_convert_h5_to_tflite import PomaDNNLinearConvertH5toTflite
from runTensorFlowModels.UpdrsDNNLinear.updrs_DNN_Linear_convert_h5_to_tflite import UpdrsDNNLinearConvertH5toTflite
from runTensorFlowModels.PomaDNNLinear.poma_DNN_Linear_keras_ver_tflite_inference import PomaDNNLinearInferenceTfLiteModel
from runTensorFlowModels.UpdrsDNNLinear.updrs_DNN_Linear_keras_ver_tflie_inference import UpdrsDNNLinearInferenceTfLiteModel

# Gradient Boosting Tree Model
from runTensorFlowModels.PomaGBT.poma_GBT import MakePomaGBTModel
from runTensorFlowModels.UpdrsGBT.updrs_GBT import MakeUpdrsGBTModel
from runTensorFlowModels.PomaGBT.poma_GBT_convert_pb_to_tflite import PomaGBTConvertPbtoTflite
from runTensorFlowModels.UpdrsGBT.updrs_GBT_convert_pb_to_tflite import UpdrsGBTConvertPbtoTflite


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

    return df


class Run:
    def __init__(self):
        pass

    def run_poma_dnn(self):
        """
        [Run Sequence]
        Load data-set -> Split to train/eval -> Model train and evaluation -> Save as H5
            -> Convert H5 model to TF-Lite model -> Save as TF-Lite

        :return: Model's accuracy, TF-Lite model's save name, TF-Lite model's save path
        """
        poma_dnn = MakePomaDnnModel()

        dataset_from_es = load_dataset_from_elastic_search()

        dataset_from_es = dataset_from_es[['poma_danger_3class', 'updrs_danger_3class', 'Velocity(m/s)', 'Cycle_time(s)', 'L_Cycle_time(s)',
                                           'R_Cycle_time(s)', 'L_Stride_length(m)', 'R_Stride_length(m)', 'L_Stride_per_min(Stride/m)',
                                           'R_Stride_per_min(stride/m)', 'L_Foot_vel.(m/s)', 'R_Foot_vel.(m/s)', 'L_step_time(s)', 'R_step_time(s)',
                                           'L_Step_per_min(step/m)', 'R_step_per_min(step/m)', 'L_Stance_time(s)' , 'R_Stance_time(s)',
                                           'L_swing_time(s)', 'R_Swing_time(s)', 'DLST_time(s)', 'DLST_Initial_time(s)', 'DLST_Terminal_time(s)',
                                           'L_Total(%)', 'L_In(%)', 'L_out(%)', 'L_front(%)', 'L_back(%)', 'L1(%)', 'L2(%)', 'L3(%)', 'L4(%)',
                                           'L5(%)', 'L6(%)', 'L7(%)', 'L8(%)', 'R_Total(%)', 'R_In(%)', 'R_out(%)', 'R_front(%)', 'R_back(%)',
                                           'R1(%)', 'R2(%)', 'R3(%)', 'R4(%)', 'R5(%)', 'R6(%)', 'R7(%)', 'R8(%)', 'L1 Balance_Time', 'L2', 'L3',
                                           'L4', 'L5', 'L6', 'L7', 'L8', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'L1-1 Sequence',
                                           'L1-2 Sequence', 'L2-1', 'L2-2', 'L3-1', 'L3-2', 'L4-1', 'L4-2', 'L5-1', 'L5-2', 'L6-1', 'L6-2', 'L7-1',
                                           'L7-2', 'L8-1', 'L8-2', 'R1-1', 'R1-2', 'R2-1', 'R2-2', 'R3-1', 'R3-2', 'R4-1', 'R4-2', 'R5-1', 'R5-2',
                                           'R6-1', 'R6-2', 'R7-1', 'R7-2', 'R8-1', 'R8-2']]

        print(dataset_from_es)

        x_train_scaled, x_eval_scaled, x_test_scaled, y_train_encoding, y_eval_encoding, y_test_encoding, labels_test = \
            poma_dnn.load_dataset(dataset_from_es=dataset_from_es)

        # save dnn models to H5 file
        h5_models_name, h5_models_path = poma_dnn.model_train_test_save(x_train_scaled=x_train_scaled,
                                                                        x_eval_scaled=x_eval_scaled,
                                                                        x_test_scaled=x_test_scaled,
                                                                        y_train_encoding=y_train_encoding,
                                                                        y_eval_encoding=y_eval_encoding,
                                                                        y_test_encoding=y_test_encoding,
                                                                        labels_test=labels_test)

        poma_dnn_convert_tflite = PomaDNNConvertH5toTflite(h5_model_name=h5_models_name, h5_model_path=h5_models_path)

        # convert dnn model H5 to Tf-lite and save
        poma_dnn_tflite_save_name, poma_dnn_tflite_save_path = poma_dnn_convert_tflite.convert_save_models()

        # Inference test TF-Lite model with random sampling dataset
        poma_dnn_inference_test = PomaDNNInferenceTfLiteModel()

        poma_x, poma_y = poma_dnn_inference_test.poma_load_dataset(test_dataset=dataset_from_es)

        accuracy = poma_dnn_inference_test.tf_lite_inference(model_path=poma_dnn_tflite_save_path, inputs=poma_x, labels=poma_y)

        return accuracy, poma_dnn_tflite_save_name, poma_dnn_tflite_save_path

    def run_updrs_dnn(self):
        """
        [Run Sequence]
        Load data-set -> Split to train/eval -> Model train and evaluation -> Save as H5
            -> Convert H5 model to TF-Lite model -> Save as TF-Lite

        :return: Model's accuracy, TF-Lite model's save name, TF-Lite model's save path
        """
        updrs_dnn = MakeUpdrsDnnModel()

        dataset_from_es = load_dataset_from_elastic_search()

        dataset_from_es = dataset_from_es[['poma_danger_3class', 'updrs_danger_3class', 'Velocity(m/s)', 'Cycle_time(s)', 'L_Cycle_time(s)',
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

        x_train_scaled, x_eval_scaled, x_test_scaled, y_train_encoding, y_eval_encoding, y_test_encoding, labels_test = \
            updrs_dnn.load_dataset(dataset_from_es=dataset_from_es)

        # save dnn models to H5 file
        h5_models_name, h5_models_path = updrs_dnn.model_train_test_save(x_train_scaled=x_train_scaled,
                                                                         x_eval_scaled=x_eval_scaled,
                                                                         x_test_scaled=x_test_scaled,
                                                                         y_train_encoding=y_train_encoding,
                                                                         y_eval_encoding=y_eval_encoding,
                                                                         y_test_encoding=y_test_encoding,
                                                                         labels_test=labels_test)

        updrs_dnn_convert_tflite = UpdrsDNNConvertH5toTflite(h5_model_name=h5_models_name, h5_model_path=h5_models_path)

        # convert dnn model H5 to Tf-lite and save
        updrs_dnn_tflite_save_name, updrs_dnn_tflite_save_path = updrs_dnn_convert_tflite.convert_save_models()

        # Inference test TF-Lite model with random sampling dataset
        updrs_dnn_inference_test = UpdrsDNNInferenceTfLiteModel()

        updrs_x, updrs_y = updrs_dnn_inference_test.updrs_load_dataset(test_dataset=dataset_from_es)

        accuracy = updrs_dnn_inference_test.tf_lite_inference(model_path=updrs_dnn_tflite_save_path, inputs=updrs_x,
                                                              labels=updrs_y)

        return accuracy, updrs_dnn_tflite_save_name, updrs_dnn_tflite_save_path

    def run_poma_dnn_linear(self):
        """
        [Run Sequence]
        Load data-set -> Split to train/eval -> Model train and evaluation -> Save as H5
            -> Convert H5 model to TF-Lite model -> Save as TF-Lite

        :return: Model's accuracy, TF-Lite model's save name, TF-Lite model's save path
        """
        poma_dnn_linear = MakePomaDnnLinearModel()

        dataset_from_es = load_dataset_from_elastic_search()

        dataset_from_es = dataset_from_es[['poma_danger_3class', 'updrs_danger_3class', 'Velocity(m/s)', 'Cycle_time(s)', 'L_Cycle_time(s)',
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

        x_train_scaled, x_eval_scaled, x_test_scaled, y_train_encoding, y_eval_encoding, y_test_encoding = \
            poma_dnn_linear.load_dataset(dataset_from_es=dataset_from_es)

        # save dnn models to H5 file
        h5_models_name, h5_models_path = poma_dnn_linear.model_train_test_save(x_train_scaled=x_train_scaled,
                                                                               x_eval_scaled=x_eval_scaled,
                                                                               x_test_scaled=x_test_scaled,
                                                                               y_train_encoding=y_train_encoding,
                                                                               y_eval_encoding=y_eval_encoding,
                                                                               y_test_encoding=y_test_encoding)

        poma_dnn_linear_convert_tflite = PomaDNNLinearConvertH5toTflite(h5_model_name=h5_models_name,
                                                                        h5_model_path=h5_models_path)

        # convert dnn model H5 to Tf-lite and save
        poma_dnn_linear_tflite_save_name, poma_dnn_linear_tflite_save_path = \
            poma_dnn_linear_convert_tflite.convert_save_models()

        # Inference test TF-Lite model with random sampling dataset
        poma_dnn_linear_inference_test = PomaDNNLinearInferenceTfLiteModel()

        poma_x, poma_y = poma_dnn_linear_inference_test.poma_load_dataset(test_dataset=dataset_from_es)

        accuracy = poma_dnn_linear_inference_test.tf_lite_inference(model_path=poma_dnn_linear_tflite_save_path,
                                                                    inputs=poma_x, labels=poma_y)

        return accuracy, poma_dnn_linear_tflite_save_name, poma_dnn_linear_tflite_save_path

    def run_updrs_dnn_linear(self):
        """
        [Run Sequence]
        Load data-set -> Split to train/eval -> Model train and evaluation -> Save as H5
            -> Convert H5 model to TF-Lite model -> Save as TF-Lite

        :return: Model's accuracy, TF-Lite model's save name, TF-Lite model's save path
        """
        updrs_dnn_linear = MakeUpdrsDnnLinearModel()

        dataset_from_es = load_dataset_from_elastic_search()

        dataset_from_es = dataset_from_es[['poma_danger_3class', 'updrs_danger_3class', 'Velocity(m/s)', 'Cycle_time(s)', 'L_Cycle_time(s)',
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

        x_train_scaled, x_eval_scaled, x_test_scaled, y_train_encoding, y_eval_encoding, y_test_encoding = \
            updrs_dnn_linear.load_dataset(dataset_from_es=dataset_from_es)

        # save dnn models to H5 file
        h5_models_name, h5_models_path = updrs_dnn_linear.model_train_test_save(x_train_scaled=x_train_scaled,
                                                                                x_eval_scaled=x_eval_scaled,
                                                                                x_test_scaled=x_test_scaled,
                                                                                y_train_encoding=y_train_encoding,
                                                                                y_eval_encoding=y_eval_encoding,
                                                                                y_test_encoding=y_test_encoding)

        updrs_dnn_linear_convert_tflite = UpdrsDNNLinearConvertH5toTflite(h5_model_name=h5_models_name,
                                                                          h5_model_path=h5_models_path)

        # convert dnn model H5 to Tf-lite and save
        updrs_dnn_linear_tflite_save_name, updrs_dnn_linear_tflite_save_path = \
            updrs_dnn_linear_convert_tflite.convert_save_models()

        # Inference test TF-Lite model with random sampling dataset
        updrs_dnn_linear_inference_test = UpdrsDNNLinearInferenceTfLiteModel()

        updrs_x, updrs_y = updrs_dnn_linear_inference_test.updrs_load_dataset(test_dataset=dataset_from_es)

        accuracy = updrs_dnn_linear_inference_test.tf_lite_inference(model_path=updrs_dnn_linear_tflite_save_path,
                                                                     inputs=updrs_x, labels=updrs_y)

        return accuracy, updrs_dnn_linear_tflite_save_name, updrs_dnn_linear_tflite_save_path

    def run_poma_gradient_boosting(self):
        poma_gbt = MakePomaGBTModel()

        dataset_from_es = load_dataset_from_elastic_search()

        dataset_from_es = dataset_from_es[['poma_danger_3class', 'updrs_danger_3class', 'Velocity(m/s)', 'Cycle_time(s)', 'L_Cycle_time(s)',
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

        x_train_scaled, x_test_scaled, labels_train, labels_test = poma_gbt.load_dataset(
            dataset_from_es=dataset_from_es)

        # save gbt models to Pb file
        pb_models_name, pb_models_path = poma_gbt.model_train_test_save(x_train_scaled=x_train_scaled,
                                                                        labels_train=labels_train,
                                                                        x_test_scaled=x_test_scaled,
                                                                        labels_test=labels_test)

        # convert gbt model Pb to Tf-lite and save
        poma_gbt_convert_tflite = PomaGBTConvertPbtoTflite(pb_model_name=pb_models_name, pb_model_path=pb_models_path)

        poma_gbt_tflite_save_name, poma_gbt_tflite_save_path = poma_gbt_convert_tflite.convert_save_models()

        return poma_gbt_tflite_save_name, poma_gbt_tflite_save_path

    def run_updrs_gradient_boosting(self):
        updrs_gbt = MakeUpdrsGBTModel()

        dataset_from_es = load_dataset_from_elastic_search()

        dataset_from_es = dataset_from_es[['poma_danger_3class', 'updrs_danger_3class', 'Velocity(m/s)', 'Cycle_time(s)', 'L_Cycle_time(s)',
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

        x_train_scaled, x_test_scaled, labels_train, labels_test = updrs_gbt.load_dataset(
            dataset_from_es=dataset_from_es)

        # save gbt models to Pb file
        pb_models_name, pb_models_path = updrs_gbt.model_train_test_save(x_train_scaled=x_train_scaled,
                                                                         labels_train=labels_train,
                                                                         x_test_scaled=x_test_scaled,
                                                                         labels_test=labels_test)

        # convert gbt model Pb to Tf-lite and save
        updrs_gbt_convert_tflite = UpdrsGBTConvertPbtoTflite(pb_model_name=pb_models_name, pb_model_path=pb_models_path)

        updrs_gbt_tflite_save_name, updrs_gbt_tflite_save_path = updrs_gbt_convert_tflite.convert_save_models()

        return updrs_gbt_tflite_save_name, updrs_gbt_tflite_save_path
