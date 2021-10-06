import datetime
import tensorflow as tf


class PomaGBTConvertPbtoTflite:
    def __init__(self, pb_model_name, pb_model_path):
        print('Start Convert and Save GBT TF-LITE Model with POMA DataSet')
        self.pb_model_name = pb_model_name
        self.pb_model_path = pb_model_path

    def convert_save_models(self):
        saved_model_obj = tf.saved_model.load(self.pb_model_path)
        print(saved_model_obj.signatures)

        # Convert the model
        concrete_func = saved_model_obj.signatures['predict']
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

        converter.experimental_new_converter = True

        # For gradient boosting models, the 'supported_ops' and 'custom_ops' operations must be enabled.
        # In native version, Boosted Trees model supports transformation x
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
        ]

        # allow_custom_ops: 사용자 지정 작업을 허용할지 여부를 나타내는 부울
        converter.allow_custom_ops = True

        # 동적 범위 양자화. 훈련 후 양자화의 가장 간단한 형태. 8 비트의 정밀도를 가진 부동 소수점에서 정수까지 가중치만 정적으로 양자화.
        # 기본 최적화를 사용하도록 optimizations 플래그를 지정
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        # Save the model.
        try:
            tflite_models_path = '/home/aiteam/daeho/PomaUpdrs/TFLite_models/GBT_TFLite_models/poma_GBT_TFLite_Model/'

            model_make_date = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
            model_name = f'poma_gbt_tflite_{model_make_date}.tflite'

            save_path = tflite_models_path + model_name

            # Save the model.
            open(save_path, 'wb').write(tflite_model)

            return model_name, save_path

        except Exception as e:
            return e

