import os

import tensorflow as tf
from tensorflow.python.keras.models import load_model

# 현재 파일 실제 경로
print(os.path.realpath(__file__))
# 현재 파일 절대 경로
print(os.path.abspath(__file__))
####################################################################################################################
# path = "/home/aiteam/daeho/PomaUpdrs/runTensorFlowModels/run/checkpoint-epoch-1000-batch-64-trial-001.h5"
#
# saved_model_obj = tf.saved_model.load(path)
# print(saved_model_obj.signatures)
# # _SignatureMap({'serving_default': <ConcreteFunction signature_wrapper(*, dense_input) at 0x7F5DEE3580B8>})

# Saved model load.
model = load_model('/home/aiteam/daeho/PomaUpdrs/runTensorFlowModels/run/checkpoint-epoch-1000-batch-64-trial-001.h5')

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# converter.target_spec.supported_ops = [
#     tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
#     tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
# ]

# # 동적 범위 양자화. 훈련 후 양자화의 가장 간단한 형태. 8 비트의 정밀도를 가진 부동 소수점에서 정수까지 가중치만 정적으로 양자화.
# # 기본 최적화를 사용하도록 optimizations 플래그를 지정
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save the model.
open("POMA_TEST_KERAS_TELITE_FILE_211112_00001.tflite", "wb").write(tflite_model)

# tensorflow.lite의 get_tensor_details 함수를 이용해서 모델 정보를 확인
interpreter = tf.lite.Interpreter(
    model_path="POMA_TEST_KERAS_TELITE_FILE_211112_00001.tflite")

for item in interpreter.get_tensor_details():
    for key in item.keys():
        print("%s : %s" % (key, item[key]))
    print("")

input_type = interpreter.get_input_details()[0]['dtype']
input_shape = interpreter.get_input_details()[0]['shape']
input_index = interpreter.get_input_details()[0]['index']
print('input: ', input_type, input_shape, input_index)

output_type = interpreter.get_output_details()[0]['dtype']
output_shape = interpreter.get_output_details()[0]['shape']
output_index = interpreter.get_output_details()[0]['index']
print('output: ', output_type, output_shape, output_index)
