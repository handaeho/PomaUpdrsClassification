import tensorflow as tf

from os import path, listdir
from shutil import copytree
from tensorflow import estimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# print('tensorflow version:', tf.__version__)
tf.random.set_seed(210813)

model_dir = '/home/aiteam/daeho/PomaUpdrs/H5_Pb_models/GBT_PB_models/updrs_GBT_PB_Model/'  # 모델을 저장할 디렉토리
model_name = 'updrs_gradient_boosting_model'  # 모델명
num_epochs = 20  # Epoch (입력 데이터를 몇회 순환할지)


def get_abs_directory_list(in_path):
    """입력된 경로의 (절대경로)디렉토리 목록을 가져옵니다"""
    out_paths = []
    out_ids = []
    if path.exists(in_path) and len(listdir(in_path)) >= 1:
        for id in listdir(in_path):
            abs_dir = path.join(in_path, id)
            if path.isdir(abs_dir):
                out_paths.append(abs_dir)
                out_ids.append(id)

    return out_paths, out_ids


def serving_input_receiver_fn():
    """Build the serving inputs."""
    # The outer dimension (None) allows us to batch up inputs for efficiency.
    # However, it also means that if we want a prediction
    # for a single instance, we'll need to wrap it in an outer list.

    receiver_tensors = {
        'RStrideperminstridem0': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R711': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R422': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R823': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L1BalanceTime4': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R35': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'DLSTTerminaltimes6': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'Rstepperminstepm7': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'RStancetimes8': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L429': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R610': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'LStancetimes11': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R312': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'DLSTInitialtimes13': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'Lout14': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L115': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'LCycletimes16': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'RIn17': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L8218': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L719': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R820': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R221': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'Lswingtimes22': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L723': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L7224': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'Rout25': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L526': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R3227': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'Rsteptimes28': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L6229': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R2130': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R8131': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'RStridelengthm32': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L8133': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L834': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L2235': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'Lsteptimes36': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L637': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L7138': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'Rback39': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L11Sequence40': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'DLSTtimes41': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L542': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L3143': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R144': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R445': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R746': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R4147': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L2148': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L349': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L650': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L251': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L852': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R5153': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'RFootvelms54': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R555': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R7256': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L357': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'Cycletimes58': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'LStrideperminStridem59': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'LFootvelms60': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'LStepperminstepm61': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'LIn62': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R263': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R564': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'Rfront65': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L466': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R167': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L12Sequence68': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'RCycletimes69': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'Lback70': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L5271': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L272': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'Velocityms73': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R774': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R6175': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R676': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R1277': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L6178': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L479': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R480': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L4181': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R5282': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'Lfront83': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L3284': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R6285': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'LStridelengthm86': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R1187': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R888': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R2289': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'RSwingtimes90': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'LTotal91': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'R3192': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'L5193': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
        'RTotal94': tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32),
    }

    features = {
        'x': tf.concat([
            receiver_tensors['RStrideperminstridem0'],
            receiver_tensors['R711'],
            receiver_tensors['R422'],
            receiver_tensors['R823'],
            receiver_tensors['L1BalanceTime4'],
            receiver_tensors['R35'],
            receiver_tensors['DLSTTerminaltimes6'],
            receiver_tensors['Rstepperminstepm7'],
            receiver_tensors['RStancetimes8'],
            receiver_tensors['L429'],
            receiver_tensors['R610'],
            receiver_tensors['LStancetimes11'],
            receiver_tensors['R312'],
            receiver_tensors['DLSTInitialtimes13'],
            receiver_tensors['Lout14'],
            receiver_tensors['L115'],
            receiver_tensors['LCycletimes16'],
            receiver_tensors['RIn17'],
            receiver_tensors['L8218'],
            receiver_tensors['L719'],
            receiver_tensors['R820'],
            receiver_tensors['R221'],
            receiver_tensors['Lswingtimes22'],
            receiver_tensors['L723'],
            receiver_tensors['L7224'],
            receiver_tensors['Rout25'],
            receiver_tensors['L526'],
            receiver_tensors['R3227'],
            receiver_tensors['Rsteptimes28'],
            receiver_tensors['L6229'],
            receiver_tensors['R2130'],
            receiver_tensors['R8131'],
            receiver_tensors['RStridelengthm32'],
            receiver_tensors['L8133'],
            receiver_tensors['L834'],
            receiver_tensors['L2235'],
            receiver_tensors['Lsteptimes36'],
            receiver_tensors['L637'],
            receiver_tensors['L7138'],
            receiver_tensors['Rback39'],
            receiver_tensors['L11Sequence40'],
            receiver_tensors['DLSTtimes41'],
            receiver_tensors['L542'],
            receiver_tensors['L3143'],
            receiver_tensors['R144'],
            receiver_tensors['R445'],
            receiver_tensors['R746'],
            receiver_tensors['R4147'],
            receiver_tensors['L2148'],
            receiver_tensors['L349'],
            receiver_tensors['L650'],
            receiver_tensors['L251'],
            receiver_tensors['L852'],
            receiver_tensors['R5153'],
            receiver_tensors['RFootvelms54'],
            receiver_tensors['R555'],
            receiver_tensors['R7256'],
            receiver_tensors['L357'],
            receiver_tensors['Cycletimes58'],
            receiver_tensors['LStrideperminStridem59'],
            receiver_tensors['LFootvelms60'],
            receiver_tensors['LStepperminstepm61'],
            receiver_tensors['LIn62'],
            receiver_tensors['R263'],
            receiver_tensors['R564'],
            receiver_tensors['Rfront65'],
            receiver_tensors['L466'],
            receiver_tensors['R167'],
            receiver_tensors['L12Sequence68'],
            receiver_tensors['RCycletimes69'],
            receiver_tensors['Lback70'],
            receiver_tensors['L5271'],
            receiver_tensors['L272'],
            receiver_tensors['Velocityms73'],
            receiver_tensors['R774'],
            receiver_tensors['R6175'],
            receiver_tensors['R676'],
            receiver_tensors['R1277'],
            receiver_tensors['L6178'],
            receiver_tensors['L479'],
            receiver_tensors['R480'],
            receiver_tensors['L4181'],
            receiver_tensors['R5282'],
            receiver_tensors['Lfront83'],
            receiver_tensors['L3284'],
            receiver_tensors['R6285'],
            receiver_tensors['LStridelengthm86'],
            receiver_tensors['R1187'],
            receiver_tensors['R888'],
            receiver_tensors['R2289'],
            receiver_tensors['RSwingtimes90'],
            receiver_tensors['LTotal91'],
            receiver_tensors['R3192'],
            receiver_tensors['L5193'],
            receiver_tensors['RTotal94']
        ], axis=1)
    }

    return tf.estimator.export.ServingInputReceiver(receiver_tensors=receiver_tensors, features=features)


class MakeUpdrsGBTModel:
    def __init__(self):
        print('Start Train and Save GBT Model with UPDRS DataSet')

    def load_dataset(self, dataset_from_es):
        """
        Method to train test split and scale data loaded from Elastic-Search.

        :param dataset_from_es: DataSet from ES (DataFrame)

        :return: Scaled train set, eval set / encoded train set labels, eval set labels
        """
        dataset_3class = dataset_from_es.copy()

        # Drop rows containing nan values from dataset
        dataset_3class.dropna(axis=0, inplace=True)

        # 필요없는 'poma_danger_3class' 컬럼은 삭제
        dataset_3class.drop(['poma_danger_3class'], axis=1, inplace=True)

        # 'updrs_danger_3class' 컬럼 pop -> 라벨
        labels = dataset_3class.pop('updrs_danger_3class').values

        # split dataset to raw train set, eval set.
        features_train, features_test, labels_train, labels_test = train_test_split(dataset_3class, labels,
                                                                                    test_size=0.2, shuffle=True,
                                                                                    random_state=1220)
        # 변형 객체 생성
        rs_scaler = RobustScaler()

        # 훈련 데이터 스케일링
        x_train_scaled = rs_scaler.fit_transform(features_train)

        # 테스트 데이터의 스케일링
        x_test_scaled = rs_scaler.transform(features_test)

        return x_train_scaled, x_test_scaled, labels_train, labels_test

    def model_train_test_save(self, x_train_scaled, x_test_scaled, labels_train, labels_test):
        train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(x={'x': x_train_scaled}, y=labels_train,
                                                                      num_epochs=num_epochs, shuffle=True)

        train_spec = estimator.TrainSpec(input_fn=train_input_fn)

        test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(x={'x': x_test_scaled}, y=labels_test,
                                                                     num_epochs=1, shuffle=False)

        ##############################################################################################################

        # 최근 모델을 저장할 Exporter
        latest_exporter = estimator.LatestExporter(name='latest_exporter',
                                                   serving_input_receiver_fn=serving_input_receiver_fn)
        # 가장 좋은 모델을 저장할 Exporter
        best_exporter = estimator.BestExporter(name='best_exporter',
                                               serving_input_receiver_fn=serving_input_receiver_fn)

        exporters = [latest_exporter, best_exporter]

        eval_spec = estimator.EvalSpec(input_fn=test_input_fn, throttle_secs=10, exporters=exporters)

        gbt_estimator = tf.estimator.BoostedTreesClassifier(config=estimator.RunConfig(model_dir=model_dir),
                                                            feature_columns=[
                                                                tf.feature_column.numeric_column('x', shape=[95])],
                                                            n_trees=300, max_depth=5, n_batches_per_layer=32,
                                                            n_classes=3, learning_rate=0.04)

        tf.estimator.train_and_evaluate(gbt_estimator, train_spec=train_spec, eval_spec=eval_spec)

        # 가장 좋은 모델이 저장된 디렉토리
        best_exporter_path = path.join(model_dir, 'export', 'best_exporter')
        src_paths, src_ids = get_abs_directory_list(best_exporter_path)

        # 서빙되고 있는 모델이 저장된 디렉토리
        serving_exporter_path = path.join(model_dir, 'export', 'serving_exporter', model_name)
        _, des_ids = get_abs_directory_list(serving_exporter_path)

        try:
            # 순회
            for idx, src_path in enumerate(src_paths):
                # 신규 모델이라면
                if src_ids[idx] not in des_ids:
                    # 복사한다
                    copytree(src_path, path.join(serving_exporter_path, src_ids[idx]))
                    print(str(src_ids[idx]) + ' copy!')

                    save_path = model_dir + 'export/' + 'best_exporter/' + src_ids[idx]

                    return model_name, save_path

        except Exception as e:
            return e

