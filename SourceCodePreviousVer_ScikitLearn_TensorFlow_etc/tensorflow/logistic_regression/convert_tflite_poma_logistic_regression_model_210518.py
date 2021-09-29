import tensorflow as tf
import tensorflow_model_optimization as tfmot

"""
tf.lite.TFLiteConverter
    - allow_custom_ops	
        사용자 지정 작업을 허용할지 여부를 나타내는 부울입니다. 
        False이면 알 수 없는 작업이 오류입니다. 
        True이면 알 수 없는 작업에 대한 사용자 지정 작업이 생성됩니다. 
        개발자는 커스텀 리졸버를 사용하여 TensorFlow Lite 런타임에이를 제공해야합니다. (기본값 False)
    - optimizations	
        모델을 변환 할 때 적용 할 최적화 목록입니다. 예 [Optimize.DEFAULT]
    - representative_dataset	
        모델에 대한 입력 및 출력 샘플을 생성하는 데 사용할 수 있는 대표적인 데이터 세트입니다. 
        변환기는 데이터 세트를 사용하여 다양한 최적화를 평가할 수 있습니다. 
        이것은 선택적 속성이지만 INT8이 대상 작업에서 유일한 지원 내장 작업 인 경우 필요합니다.
    - target_spec	
        대상 장치의 사양.
    - inference_input_type	
        입력 계층의 데이터 유형입니다. 
        정수 유형 (tf.int8 및 tf.uint8)은 현재 학습 후 정수 양자화 및 양자화 인식 교육에만 지원됩니다. 
        (기본값 tf.float32, {tf.float32, tf.int8, tf.uint8}에 있어야 함)
    - inference_output_type	  
        출력 계층의 데이터 유형입니다.
        정수 유형 (tf.int8 및 tf.uint8)은 현재 학습 후 정수 양자화 및 양자화 인식 교육에만 지원됩니다. 
        (기본값 tf.float32, {tf.float32, tf.int8, tf.uint8}에 있어야 함)
    - experimental_new_converter	
        TOCO 변환 대신 MLIR 기반 변환을 사용합니다. (기본값 True)


    >> convert()
        인스턴스 변수를 기반으로 TensorFlow GraphDef를 변환합니다.

        보고
          직렬화 된 형식으로 변환 된 데이터입니다.
        raise
          ValueError	
            구체적인 기능이 지정되지 않았습니다. 여러 구체적인 기능이 지정됩니다. 입력 형태가 지정되지 않았습니다. 잘못된 양자화 매개 변수입니다.

    >> from_concrete_functions

        @classmethod
        from_concrete_functions(
            funcs
        )

        ConcreteFunctions에서 TFLiteConverter 객체를 만듭니다.

        Args
          funcs	
            TensorFlow ConcreteFunctions 목록. 목록에 중복 요소가 없어야합니다. 현재 변환기는 하나의 ConcreteFunction 만 변환 할 수 있습니다.
        보고
          TFLiteConverter 객체.
        raise
          잘못된 입력 유형입니다.

    >> from_keras_model

        @classmethod
        from_keras_model(
            model
        )

        Keras 모델에서 TFLiteConverter 개체를 만듭니다.

        Args
          model	
            tf.Hard.Model
        보고
          TFLiteConverter 객체.

    >> from_saved_model

        @classmethod
        from_saved_model(
            saved_model_dir, signature_keys=None, tags=None
        )

        SavedModel 디렉터리에서 TFLiteConverter 개체를 만듭니다.

        Args
          saved_model_dir	
            변환 할 저장된 모델 디렉토리입니다.
          signature_keys	
            입력 및 출력을 포함하는 SignatureDef를 식별하는 키 목록입니다. 요소는 중복되지 않아야합니다. 
            기본적으로 signaturesMetaGraphdef 의 속성이 사용됩니다. (기본 saved_model.signatures)
          tags	
            분석 할 저장된 모델 내에서 MetaGraphDef를 식별하는 태그 세트입니다. 태그 세트의 모든 태그가 있어야합니다. (기본 설정 (SERVING))
        보고
          TFLiteConverter 객체.
        레이즈
          잘못된 서명 키입니다.
"""


path = "new_poma_logistic_regression_model_210518/1621904373"

saved_model_obj = tf.saved_model.load(path)
print(saved_model_obj.signatures)
# -> _SignatureMap({'classification': <ConcreteFunction pruned(inputs) at 0x7FA66C2E4F60>,
#                   'predict': <ConcreteFunction pruned(examples) at 0x7FA654697080>,
#                   'serving_default': <ConcreteFunction pruned(inputs) at 0x7FA6545380B8>})

# Convert the model
concrete_func = saved_model_obj.signatures['serving_default']
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

# 동적 범위 양자화: 8비트까지의 범위를 기반으로 활성화 값 등을 동적으로 양자화 하고 8비트 weights & bias 등으로 계산을 수행
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]

# allow_custom_ops: 사용자 지정 작업을 허용할지 여부를 나타내는 부울
converter.allow_custom_ops = True

# # experimental_new_converter: TOCO 변환 대신 MLIR 기반 변환을 사용
# converter.experimental_new_converter = True

tflite_model = converter.convert()

# Save the model.
open("new_poma_logistic_regression_210527.tflite", "wb").write(tflite_model)

# tensorflow.lite의 get_tensor_details 함수를 이용해서 모델 정보를 확인
interpreter = tf.lite.Interpreter(model_path="new_poma_logistic_regression_210527.tflite")

for item in interpreter.get_tensor_details():
    for key in item.keys():
        print("%s : %s" % (key, item[key]))
    print("")


# # float 16으로 양자화: weights 등을 16비트 부동 소수점 표준인 float16으로 양자화.
# # 모든 가중치가 절반의 크기를 가지며 정확도 손실을 최소하 할수 있지만, CPU 에서 실행될 때 이는 float32로 가중치 값이 역 양자화 된다.
# converter.target_spec.supported_types = [tf.float16]

# # 모델 가지치기
# pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
#                         initial_sparsity=0.0, final_sparsity=0.5,
#                         begin_step=2000, end_step=4000)
#
# model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(saved_model_obj, pruning_schedule=pruning_schedule)


# # Convert the model
# converter = tf.lite.TFLiteConverter.from_saved_model(path, signature_keys='serving_default')
# tflite_model = converter.convert()
#
# # Save the model.
# with open('updrs_dnn_210426.tflite', 'wb') as f:
#     f.write(tflite_model)


