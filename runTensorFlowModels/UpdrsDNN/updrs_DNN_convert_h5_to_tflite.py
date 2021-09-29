import datetime
import tensorflow as tf

from tensorflow.python.keras.models import load_model


class UpdrsDNNConvertH5toTflite:
    def __init__(self, h5_model_name, h5_model_path):
        print('Start Convert and Save DNN TF-LITE Model with UPDRS DataSet')
        self.h5_model_name = h5_model_name
        self.h5_model_path = h5_model_path

    def convert_save_models(self):
        """
        Convert saved H5 Model to TF-Lite model and save.

        :return: model name, save path
        """
        # Saved model load.
        model = load_model(self.h5_model_path)

        # Convert the model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # 동적 범위 양자화. 훈련 후 양자화의 가장 간단한 형태. 8 비트의 정밀도를 가진 부동 소수점에서 정수까지 가중치만 정적으로 양자화.
        # 기본 최적화를 사용하도록 optimizations 플래그를 지정
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        # save
        try:
            tflite_models_path = '/home/aiteam/daeho/PomaUpdrs/TFLite_models/DNN_TFLite_models/updrs_DNN_TFLite_Model/'

            model_make_date = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
            model_name = f'updrs_dnn_tflite_{model_make_date}.tflite'

            save_path = tflite_models_path + model_name

            # Save the model.
            open(save_path, 'wb').write(tflite_model)

            return model_name, save_path

        except Exception as e:
            return e
