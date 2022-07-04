# !/home/aiteam/.conda/envs/daehoPython38/bin/python
"""
2022.05.31 Han Dae Ho

This is about the changed logic.

The logic for receiving data from 3L-labs, running the model, and sending results to 3L-labs.

received data: String(95 columns, delimiter is 'LF')
result: danger class(e.g. 0(normal), 1(caution), 2(danger))

logic: received data. -> pretreatment. -> predict. -> send result.

result of data to be stored in ES: each row's data and each row's result.
results to be sent(but only sending, not saving): danger class
"""
import datetime
import pandas as pd
import sys
import os
import numpy as np

from datetime import timedelta
from flask import Flask, request, session
from flask_cors import CORS
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# import top-level parent and parent folder paths (append to system path) -> /home/aiteam/daeho/PomaUpdrs
sys.path.append((os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))))
print(sys.path)

# DNN Model
from runTensorFlowModels.PomaDNN.poma_DNN_220531 import MakePomaDnnModel
from runTensorFlowModels.UpdrsDNN.updrs_DNN_220531 import MakeUpdrsDnnModel


# Flask
app = Flask(__name__)
CORS(app, resources={r'*': {'origins': '*'}})
app.config['CORS_HEADERS'] = 'Content-Type'

# # ES
# es = Elasticsearch(host='192.168.0.173', port='9200')
# # print(es.info())
#
# es.indices.create(index='bulk_test', body={})


@app.before_request
def make_session_permanent():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes=100)


@app.route('/shutdown', methods=['GET'])
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return "SERVER Shutdown..."


@app.route('/gaionPomaUpdrs/receive', methods=['POST'])
def receive_data():
    """
    data format: String

    1, 2, 3, 4, 5, 6, ..., 94, 95 \n
    1, 2, 3, 4, 5, 6, ..., 94, 95 \n
    1, 2, 3, 4, 5, 6, ..., 94, 95 \n
    ...
    1, 2, 3, 4, 5, 6, ..., 94, 95 \n
    1, 2, 3, 4, 5, 6, ..., 94, 95 \n
    1, 2, 3, 4, 5, 6, ..., 94, 95

    > received stream data as sting(binary)
    > convert each data to float and 2d list like [[1, 2, 3, ...], [1, 2, 3, ...], ... ]
    > make dataframe with float stream data list

    :return: dataframe
    """
    splitlines_data = request.stream.read().decode().splitlines()

    user_list = []
    data_list = []

    for i in splitlines_data:
        split_each_string = i.split(',')

        id_password = [str(x) for x in split_each_string[0:2]]
        split_each_float = [float(x) for x in split_each_string[2:]]

        user_list.append(id_password)
        data_list.append(split_each_float)

    # 2차원 리스트 중복 제거
    user_list = list(set([tuple(set(user)) for user in user_list]))
    user_list = np.array([list(x) for x in user_list]).flatten()
    user_id_password = sorted(np.unique(user_list))

    # 'poma_danger_3class', 'updrs_danger_3class' 자리에 ID와 PassWord가 들어옴? --> 그러면 이게 한 행 마다 id, pw가 오는건지? 아니면 id, pw는 한 번만 오고 데이터가 쭉 붙는건지?
    stream_data_dataframe = pd.DataFrame(data_list, columns=[['Velocity(m/s)', 'Cycle_time(s)', 'L_Cycle_time(s)','R_Cycle_time(s)', 'L_Stride_length(m)',
                                                              'R_Stride_length(m)', 'L_Stride_per_min(Stride/m)', 'R_Stride_per_min(stride/m)', 'L_Foot_vel.(m/s)',
                                                              'R_Foot_vel.(m/s)', 'L_step_time(s)', 'R_step_time(s)', 'L_Step_per_min(step/m)', 'R_step_per_min(step/m)',
                                                              'L_Stance_time(s)', 'R_Stance_time(s)', 'L_swing_time(s)', 'R_Swing_time(s)', 'DLST_time(s)', 'DLST_Initial_time(s)',
                                                              'DLST_Terminal_time(s)', 'L_Total(%)', 'L_In(%)', 'L_out(%)', 'L_front(%)', 'L_back(%)', 'L1(%)', 'L2(%)', 'L3(%)', 'L4(%)',
                                                              'L5(%)', 'L6(%)', 'L7(%)', 'L8(%)', 'R_Total(%)', 'R_In(%)', 'R_out(%)', 'R_front(%)', 'R_back(%)',
                                                              'R1(%)', 'R2(%)', 'R3(%)', 'R4(%)', 'R5(%)', 'R6(%)', 'R7(%)', 'R8(%)', 'L1 Balance_Time', 'L2', 'L3',
                                                              'L4', 'L5', 'L6', 'L7', 'L8', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'L1-1 Sequence',
                                                              'L1-2 Sequence', 'L2-1', 'L2-2', 'L3-1', 'L3-2', 'L4-1', 'L4-2', 'L5-1', 'L5-2', 'L6-1', 'L6-2', 'L7-1',
                                                              'L7-2', 'L8-1', 'L8-2', 'R1-1', 'R1-2', 'R2-1', 'R2-2', 'R3-1', 'R3-2', 'R4-1', 'R4-2', 'R5-1', 'R5-2',
                                                              'R6-1', 'R6-2', 'R7-1', 'R7-2', 'R8-1', 'R8-2']])
    # print(stream_data_dataframe)

    return user_id_password, stream_data_dataframe


@app.route('/gaionPomaUpdrs/getPoma', methods=['POST'])
def run_poma_dnn():
    """
    [Run Sequence]
    Received dataset load -> Saved model load -> input data -> predict -> return result class

    - this method is not train and evaluation.
    - predict the input data with the previously trained model and return the result.
    - result is returned to URL.
    - row's data and row's result are saved to ES.

    :return: Model's result, accuracy
    """
    user_id_password, dataframe = receive_data()

    print(dataframe)

    if request.method == 'POST':
        if user_id_password == ['3llabs', 'qlalfqjsgh']:
            try:
                # class build
                poma_dnn = MakePomaDnnModel(dataframe)

                scaled_input_data = poma_dnn.input_data()

                predicted_result = poma_dnn.saved_model_predict(model_name='poma_dnn_keras_ver_20211206_091024.h5', input_data=scaled_input_data)

                # position of max probability
                predictions = []
                for i in predicted_result:
                    predictions.append(np.argmax(i))

                print(predictions)

                # most elements
                max_count = max(predictions, key=predictions.count)

                dataframe['result'] = predictions

                # save to csv data(feature) + row's result
                saved_path = '/home/aiteam/daeho/PomaUpdrs/DataSet_From_3L_labs_only_features'
                dataframe.to_csv(saved_path + f'/poma_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', encoding='CP949', index=False)

                # TODO: ES에 recevied data + row's result 저장하기 (return은 X)
                # TODO: 새로운 데이터를 위한 ES index 만들기. (새로 오는 데이터는 label이 없기 떄문에 vector만 있는 저장소가 필요)

                # # insert data to ES
                # documents = dataframe.to_dict(orient='records')
                # bulk(es, documents, index='foot_logger_poma_updrs_3l_labs_data', doc_type='_doc', raise_on_error=True)

                return str(max_count)

            except Exception as E:
                print(E)

                return str(E), '-1'

        else:
            return 'Wrong User. please check ID or PassWord.'

    else:
        return 'Plz change request method to "POST" and try again'


@app.route('/gaionPomaUpdrs/getUpdrs', methods=['POST'])
def run_updrs_dnn():
    """
    [Run Sequence]
    received dataset load -> Split to train/eval -> Model train and evaluation -> predict ->

    - model is saved as H5 in server.
    - result is returned to URL.
    - row's data and row's result are saved to ES.

    :return: Model's result, accuracy
    """
    user_id_password, dataframe = receive_data()

    if request.method == 'POST':
        if user_id_password == ['3llabs', 'qlalfqjsgh']:
            try:
                # class build
                updrs_dnn = MakeUpdrsDnnModel(dataframe)

                scaled_input_data = updrs_dnn.input_data()

                predicted_result = updrs_dnn.saved_model_predict(model_name='updrs_dnn_keras_ver_20211206_091801.h5', input_data=scaled_input_data)

                # position of max probability
                predictions = []
                for i in predicted_result:
                    predictions.append(np.argmax(i))

                print(predictions)

                # most elements
                max_count = max(predictions, key=predictions.count)

                dataframe['result'] = predictions

                # save to csv data(feature) + row's result
                saved_path = '/home/aiteam/daeho/PomaUpdrs/DataSet_From_3L_labs_only_features'
                dataframe.to_csv(saved_path + f'/updrs_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', encoding='CP949', index=False)

                # TODO: ES에 recevied data + row's result 저장하기 (return은 X)
                # TODO: 새로운 데이터를 위한 ES index 만들기. (새로 오는 데이터는 label이 없기 떄문에 vector만 있는 저장소가 필요)

                return str(max_count)

            except Exception as E:
                print(E)

                return '-1'

        else:
            return 'Plz change request method to "POST" and try again'

    else:
        return 'Plz change request method to "POST" and try again'


if __name__ == '__main__':
    app.config['SESSION_TYPE'] = 'filesystem'
    app.secret_key = 'super secret key'

    app.run(host='0.0.0.0', port=5000)
