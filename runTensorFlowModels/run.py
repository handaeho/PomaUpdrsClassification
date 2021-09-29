"""
This is the code to load data, create DNN model, save H5 model, and create TF-Lite file.
"""
import os
import pandas as pd

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search

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
from runTensorFlowModels.PomaDNNLinear.poma_DNN_Linear_keras_ver_tflite_inference import \
    PomaDNNLinearInferenceTfLiteModel
from runTensorFlowModels.UpdrsDNNLinear.updrs_DNN_Linear_keras_ver_tflie_inference import \
    UpdrsDNNLinearInferenceTfLiteModel


# print(os.path.realpath(__file__))  # 현재 파일 실제 경로
# print(os.path.abspath(__file__))  # 현재 파일 절대 경로


class Run:
    def __init__(self):
        pass

    # load data from Elastic-Search.
    def load_dataset_from_elastic_search(self):
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

    def run_poma_dnn(self):
        """
        [Run Sequence]
        Load data-set -> Split to train/eval -> Model train and evaluation -> Save as H5
            -> Convert H5 model to TF-Lite model -> Save as TF-Lite

        :return: Model's accuracy, TF-Lite model's save name, TF-Lite model's save path
        """
        poma_dnn = MakePomaDnnModel()

        dataset_from_es = self.load_dataset_from_elastic_search()

        x_train_scaled, x_eval_scaled, x_test_scaled, y_train_encoding, y_eval_encoding, y_test_encoding = \
            poma_dnn.load_dataset(dataset_from_es=dataset_from_es)

        # save dnn models to H5 file
        h5_models_name, h5_models_path = poma_dnn.model_train_test_save(x_train_scaled=x_train_scaled,
                                                                        x_eval_scaled=x_eval_scaled,
                                                                        x_test_scaled=x_test_scaled,
                                                                        y_train_encoding=y_train_encoding,
                                                                        y_eval_encoding=y_eval_encoding,
                                                                        y_test_encoding=y_test_encoding)

        poma_dnn_convert_tflite = PomaDNNConvertH5toTflite(h5_model_name=h5_models_name, h5_model_path=h5_models_path)

        # convert dnn model H5 to Tf-lite and save
        poma_tflite_save_name, poma_tflite_save_path = poma_dnn_convert_tflite.convert_save_models()

        # Inference test TF-Lite model with random sampling dataset
        poma_dnn_inference_test = PomaDNNInferenceTfLiteModel()

        poma_x, poma_y = poma_dnn_inference_test.poma_load_dataset(test_dataset=dataset_from_es)

        accuracy = poma_dnn_inference_test.tf_lite_inference(model_path=poma_tflite_save_path,
                                                             inputs=poma_x, labels=poma_y)

        return accuracy, poma_tflite_save_name, poma_tflite_save_path

    def run_updrs_dnn(self):
        """
        [Run Sequence]
        Load data-set -> Split to train/eval -> Model train and evaluation -> Save as H5
            -> Convert H5 model to TF-Lite model -> Save as TF-Lite

        :return: Model's accuracy, TF-Lite model's save name, TF-Lite model's save path
        """
        updrs_dnn = MakeUpdrsDnnModel()

        dataset_from_es = self.load_dataset_from_elastic_search()

        x_train_scaled, x_eval_scaled, x_test_scaled, y_train_encoding, y_eval_encoding, y_test_encoding = \
            updrs_dnn.load_dataset(dataset_from_es=dataset_from_es)

        # save dnn models to H5 file
        h5_models_name, h5_models_path = updrs_dnn.model_train_test_save(x_train_scaled=x_train_scaled,
                                                                         x_eval_scaled=x_eval_scaled,
                                                                         x_test_scaled=x_test_scaled,
                                                                         y_train_encoding=y_train_encoding,
                                                                         y_eval_encoding=y_eval_encoding,
                                                                         y_test_encoding=y_test_encoding)

        updrs_dnn_convert_tflite = UpdrsDNNConvertH5toTflite(h5_model_name=h5_models_name, h5_model_path=h5_models_path)

        # convert dnn model H5 to Tf-lite and save
        updrs_tflite_save_name, updrs_tflite_save_path = updrs_dnn_convert_tflite.convert_save_models()

        # Inference test TF-Lite model with random sampling dataset
        updrs_dnn_inference_test = UpdrsDNNInferenceTfLiteModel()

        updrs_x, updrs_y = updrs_dnn_inference_test.updrs_load_dataset(test_dataset=dataset_from_es)

        accuracy = updrs_dnn_inference_test.tf_lite_inference(model_path=updrs_tflite_save_path, inputs=updrs_x,
                                                              labels=updrs_y)

        return accuracy, updrs_tflite_save_name, updrs_tflite_save_path

    def run_poma_dnn_linear(self):
        """
        [Run Sequence]
        Load data-set -> Split to train/eval -> Model train and evaluation -> Save as H5
            -> Convert H5 model to TF-Lite model -> Save as TF-Lite

        :return: Model's accuracy, TF-Lite model's save name, TF-Lite model's save path
        """
        poma_dnn_linear = MakePomaDnnLinearModel()

        dataset_from_es = self.load_dataset_from_elastic_search()

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
        poma_tflite_save_name, poma_tflite_save_path = poma_dnn_linear_convert_tflite.convert_save_models()

        # Inference test TF-Lite model with random sampling dataset
        poma_dnn_linear_inference_test = PomaDNNLinearInferenceTfLiteModel()

        poma_x, poma_y = poma_dnn_linear_inference_test.poma_load_dataset(test_dataset=dataset_from_es)

        accuracy = poma_dnn_linear_inference_test.tf_lite_inference(model_path=poma_tflite_save_path,
                                                                    inputs=poma_x, labels=poma_y)

        return accuracy, poma_tflite_save_name, poma_tflite_save_path

    def run_updrs_dnn_linear(self):
        """
        [Run Sequence]
        Load data-set -> Split to train/eval -> Model train and evaluation -> Save as H5
            -> Convert H5 model to TF-Lite model -> Save as TF-Lite

        :return: Model's accuracy, TF-Lite model's save name, TF-Lite model's save path
        """
        updrs_dnn_linear = MakeUpdrsDnnLinearModel()

        dataset_from_es = self.load_dataset_from_elastic_search()

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
        updrs_tflite_save_name, updrs_tflite_save_path = updrs_dnn_linear_convert_tflite.convert_save_models()

        # Inference test TF-Lite model with random sampling dataset
        updrs_dnn_linear_inference_test = UpdrsDNNLinearInferenceTfLiteModel()

        updrs_x, updrs_y = updrs_dnn_linear_inference_test.updrs_load_dataset(test_dataset=dataset_from_es)

        accuracy = updrs_dnn_linear_inference_test.tf_lite_inference(model_path=updrs_tflite_save_path,
                                                                     inputs=updrs_x, labels=updrs_y)

        return accuracy, updrs_tflite_save_name, updrs_tflite_save_path


if __name__ == '__main__':
    run = Run()

    x = run.load_dataset_from_elastic_search()
    print(x)

    print('-------------------- DNN Model start --------------------')
    poma_dnn_acc, poma_dnn_tflite_save_name, poma_dnn_tflite_save_path = run.run_poma_dnn()
    print('The POMA accuracy predicted by the DNN TF-Lite Model:', poma_dnn_acc)
    print('Model Name =>', poma_dnn_tflite_save_name, 'Model Path =>', poma_dnn_tflite_save_path)

    #

    print('----------------------------------------------------------------------------------------------------')

    updrs_dnn_acc, updrs_dnn_tflite_save_name, updrs_dnn_tflite_save_path = run.run_updrs_dnn()
    print('The UPDRS accuracy predicted by the DNN TF-Lite Model:', updrs_dnn_acc)
    print('Model Name =>', updrs_dnn_tflite_save_name, 'Model Path =>', updrs_dnn_tflite_save_path)

    #

    print('-------------------- DNN-Linear combined Model start --------------------')
    poma_dnn_linear_acc, poma_dnn_linear_tflite_save_name, poma_dnn_linear_tflite_save_path = run.run_poma_dnn_linear()
    print('The POMA accuracy predicted by the DNN-Linear combined TF-Lite Model:', poma_dnn_linear_acc)
    print('Model Name =>', poma_dnn_linear_tflite_save_name, 'Model Path =>', poma_dnn_linear_tflite_save_path)

    #

    print('----------------------------------------------------------------------------------------------------')

    updrs_dnn_linear_acc, updrs_dnn_linear_tflite_save_name, updrs_dnn_linear_tflite_save_path = run.run_updrs_dnn_linear()
    print('The UPDRS accuracy predicted by the DNN-Linear combined TF-Lite Model:', updrs_dnn_linear_acc)
    print('Model Name =>', updrs_dnn_linear_tflite_save_name, 'Model Path =>', updrs_dnn_linear_tflite_save_path)

    #


