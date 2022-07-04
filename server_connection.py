#!/home/aiteam/.conda/envs/daehoPython38/bin/python3.8
# -*- coding: utf-8 -*-

import os
import datetime

from flask import Flask, request
from werkzeug.utils import secure_filename

UPLOAD_DIR = "/home/aiteam/daeho/PomaUpdrs/DataSet_From_3L_labs"
app = Flask(__name__)
app.config['UPLOAD_DIR'] = UPLOAD_DIR


@app.route('/file-upload', methods=['POST'])
def upload_files():
    if request.method == 'POST':
        f = request.files['file']

        f_name = secure_filename(f.filename)

        path = os.path.join(app.config['UPLOAD_DIR'], f_name)

        f.save(path)

        return 'File upload complete =====> {name} / {date}'.format(name=f_name, date=datetime.datetime.now())

    else:
        return 'This is not a valid transmission method.'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)

