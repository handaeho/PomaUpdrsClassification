import pathlib
import tensorflow as tf

####################################################################################################################

path = "updrs_dnn_test_210604/export/serving_exporter/updrs_dnn_0001/1622788484"

saved_model_obj = tf.saved_model.load(path)
print(saved_model_obj.signatures)

# Convert the model
concrete_func = saved_model_obj.signatures['predict']
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]

# 동적 범위 양자화. 훈련 후 양자화의 가장 간단한 형태. 8 비트의 정밀도를 가진 부동 소수점에서 정수까지 가중치만 정적으로 양자화.
# 기본 최적화를 사용하도록 optimizations 플래그를 지정
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# # allow_custom_ops: 사용자 지정 작업을 허용할지 여부를 나타내는 부울
# converter.allow_custom_ops = True

# # experimental_new_converter: TOCO 변환 대신 MLIR 기반 변환을 사용
# converter.experimental_new_converter = True

tflite_model = converter.convert()

# Save the model.
tflite_models_dir = pathlib.Path()
tflite_models_dir.mkdir(exist_ok=True, parents=True)

tflite_model_file = tflite_models_dir / "new_updrs_dnn_model_210604_quantization_0001.tflite"
# tflite_model_file = tflite_models_dir / "new_updrs_dnn_model_210604_standard_0001.tflite"
tflite_model_file.write_bytes(tflite_model)

# tensorflow.lite의 get_tensor_details 함수를 이용해서 모델 정보를 확인
interpreter = tf.lite.Interpreter(model_path="new_updrs_dnn_model_210604_quantization_0001.tflite")
# interpreter = tf.lite.Interpreter(model_path="new_updrs_dnn_model_210604_standard_0001.tflite")

for item in interpreter.get_tensor_details():
    for key in item.keys():
        print("%s : %s" % (key, item[key]))
    print("")

input_type = interpreter.get_input_details()[0]['dtype']
input_shape = interpreter.get_input_details()[0]['shape']
print('input: ', input_type, input_shape)

output_type = interpreter.get_output_details()[0]['dtype']
output_shape = interpreter.get_output_details()[0]['shape']
print('output: ', output_type, output_shape)

