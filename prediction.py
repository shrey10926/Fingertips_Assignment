from pickle import load
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, make_response, abort
import logging
from flask_cors import CORS

log_format = "%(asctime)s::%(name)s::"\
             "%(filename)s::%(funcName)s::%(lineno)d::%(message)s"
logging.basicConfig(filename='log_file.log', filemode='w', level='DEBUG', format=log_format)
logger = logging.getLogger()


app = Flask(__name__)
CORS(app)


def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response


def custom_abort(message,code):

    if code == 200:
        json = jsonify(message)
        response = make_response(json,200)
        return abort(response)

    else:
        json = jsonify(errorMessage = message)
        response = make_response(json,400)
        return abort(response)


def separate_df(data):  
  num = data.select_dtypes(include = 'number')
  cat = data.select_dtypes(include = 'object')
  cat = cat.reset_index(drop = True)
  return num, cat


@app.route('/predict', methods = ['POST'])
def flask_api():

  if request.method == 'OPTIONS':
      return _build_cors_preflight_response()


  try:
      logger.debug('Fetching data from request...')
      data = request.get_json()
      logger.debug('Data Fetched!')

  except Exception as e:
    logger.debug(e)
    return custom_abort(str(e), 400)


  for k, v in data.items():

    try:
      # load the model
      model = load(open('model.pkl', 'rb'))
      # load the scaler
      scaler = load(open('scaler.pkl', 'rb'))
      # load the encoder
      enc = load(open('encoder.pkl', 'rb'))
    except Exception as e:
      logger.debug(e)
      return custom_abort(str(e), 400)

    try:
      test_df = pd.DataFrame([data.get('entries')], columns =['gender', 'ssc_percentage', 'ssc_board', 'hsc_percentage', 'hsc_board', 'hsc_subject', 'degree_percentage', 'undergrad_degree', 'work_experience', 'emp_test_percentage', 'specialisation', 'mba_percent']) 
    except Exception as e:
      logger.debug(e)
      return custom_abort(str(e), 400)

    try:
      num, cat = separate_df(test_df)
      test_df = pd.concat([num, cat], axis = 1)    
    except Exception as e:
      logger.debug(e)
      return custom_abort(str(e), 400)

    try:
      nutest_dfm = pd.DataFrame(enc.transform(test_df), columns = test_df.columns)
      nutest_dfm = pd.DataFrame(scaler.transform(nutest_dfm), columns = nutest_dfm.columns)
    except Exception as e:
      logger.debug(e)
      return custom_abort(str(e), 400)

    try:
      final_pred = model.predict(nutest_dfm)
    except Exception as e:
      logger.debug(e)
      return custom_abort(str(e), 400)


    if final_pred == 1:
      return 'Placed'
    else:
      return 'Not Placed'

  return custom_abort('Done!', 200)


if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug = True, port = 6996)