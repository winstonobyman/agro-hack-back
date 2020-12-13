from tensorflow import keras
import numpy as np
import pickle
import os
from hdbscan import HDBSCAN, approximate_predict
from utils import get_sensor_data



# словарь для маппинга слов
pair_dict = {
    'currentLightingLevel': 'illumination',
    'currentSoilAcidity': 'soil_acidity',
    'currentSoilMoisture':'relative_soil_mosture',
    'currentTemperature':'air_temperature'
}

scaler_path = os.path.join('ml', 'anomaly', 'scaler_lstm.pkl')
with open(scaler_path, 'rb') as f:
    scaler_lstm = pickle.load(f)

TIME_STEPS = 30

threshold_path = os.path.join('ml', 'anomaly', 'THRESHOLD.txt')
with open(threshold_path) as f:
    THRESHOLD = float(f.read().strip())

model_path = os.path.join('ml', 'anomaly', 'model_lstm.h5')
model_lstm = keras.models.load_model(model_path, compile=False)

SENSOR_PATH = os.path.join('data/sensor_data.csv')
SENSOR_DATA = get_sensor_data(SENSOR_PATH, ascending=True)


def check_last_anomaly():
    data_to_predict = check_anomalies(SENSOR_DATA['air_temperature'][-TIME_STEPS:])
    return data_to_predict.flatten()[-1]


def transform_to_predict_input(data, scaler=scaler_lstm, TIME_STEPS=TIME_STEPS):
  'Transform iterable object with lenght equals or multiplicle to TIME_STEPS for predictoin'
  dlen = len(data)
  if dlen % TIME_STEPS != 0:
    raise ValueError('Data is iterable with TIME_STEPS multiplicity lenght')
  return scaler.transform(np.array(data).reshape(1, dlen)).reshape(dlen // TIME_STEPS, TIME_STEPS, 1)


def check_anomalies(values, model=model_lstm):
  'Check value for anomalousness'

  for_prediction = transform_to_predict_input(values)
  predicted = model.predict(for_prediction)
  diff = predicted - for_prediction
  return abs(diff) > THRESHOLD


def load_clusterer(filename):
  with open(filename, 'rb') as f:
    model = pickle.load(f)
  return model


def predict_anom_for_field(fieldname, data, number, pair_dict=pair_dict):
  if data == 256:
    return True
  clusterer = load_clusterer('ml/anomaly/clusterers/' + pair_dict[fieldname]
                         + '_p' + str(number) + '.pkl')
  pred_label, strength = approximate_predict(clusterer, [[data]])
  if pred_label == -1:
    return True
  return False

def add_anomaly_to_dict(ddict):
  for i, d in enumerate(ddict['data']):
    d['isTemperatureAnomal'] = check_last_anomaly()

    d['isSoilMoistureAnomal'] = predict_anom_for_field('currentSoilMoisture',
                                                        d['currentSoilMoisture'], i + 1, pair_dict)

    d['isLightingLevelAnomal'] = predict_anom_for_field('currentLightingLevel',
                                                        d['currentLightingLevel'], i + 1, pair_dict)

    d['isSoilAcidityAnomal'] = predict_anom_for_field('currentSoilAcidity',
                                                        d['currentSoilAcidity'], i + 1, pair_dict)

  return ddict
