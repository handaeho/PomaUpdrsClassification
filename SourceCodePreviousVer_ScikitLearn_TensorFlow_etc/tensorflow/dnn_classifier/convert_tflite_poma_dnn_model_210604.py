import pathlib
import tensorflow as tf

####################################################################################################################

# path = "poma_dnn_test_210803/export/serving_exporter/poma_dnn_0001/1627976196"
path = "TEST_POMA_DNN_MODEL_0000000000000000000000000210809/1628496582"

saved_model_obj = tf.saved_model.load(path)
print(saved_model_obj.signatures)

# Convert the model
concrete_func = saved_model_obj.signatures['predict']

converter = tf.lite.TFLiteConverter.from_saved_model(path)

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]

converter.target_spec.supported_types = [tf.float32]

converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

# 동적 범위 양자화. 훈련 후 양자화의 가장 간단한 형태. 8 비트의 정밀도를 가진 부동 소수점에서 정수까지 가중치만 정적으로 양자화.
# 기본 최적화를 사용하도록 optimizations 플래그를 지정
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# allow_custom_ops: 사용자 지정 작업을 허용할지 여부를 나타내는 부울
converter.allow_custom_ops = True

# experimental_new_converter:  TOCO 변환 대신 MLIR 기반 변환을 활성화합니다. (기본값 True)
converter.experimental_new_converter = True

# Flatbuffer 기반 변환 대신 MLIR 기반 양자화 변환을 활성화합니다. (기본값 True)
converter.experimental_new_converter = False

tflite_model = converter.convert()

# Save the model.
open("TEST_TELITE_FILE_00000000000000000000000000000000003.tflite", "wb").write(tflite_model)

# # Save the model.
# tflite_models_dir = pathlib.Path()
# tflite_models_dir.mkdir(exist_ok=True, parents=True)
#
# tflite_model_file = tflite_models_dir / "TEST_TELITE_FILE_00000000000000000000000000000000003.tflite"
# # tflite_model_file = tflite_models_dir / "new_poma_dnn_model_210604_standard_0001.tflite"
# tflite_model_file.write_bytes(tflite_model)

# tensorflow.lite의 get_tensor_details 함수를 이용해서 모델 정보를 확인
interpreter = tf.lite.Interpreter(model_path="TEST_TELITE_FILE_00000000000000000000000000000000003.tflite")
# interpreter = tf.lite.Interpreter(model_path="new_poma_dnn_model_210604_standard_0001.tflite")

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



