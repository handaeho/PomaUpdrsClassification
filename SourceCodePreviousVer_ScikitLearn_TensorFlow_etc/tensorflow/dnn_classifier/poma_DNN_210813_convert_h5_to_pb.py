from tensorflow.python.keras.models import load_model

# Saved model load.
model = load_model('poma_dnn_keras_ver_210813_00000000000000000000000000000001.h5', compile=False)

export_path = 'poma_DNN_TEST_h5_to_pb_210813_00000000000000000000000000000001'
model.save(export_path, save_format="tf")
