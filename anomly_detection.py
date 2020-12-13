from tensorflow import keras
import numpy as np
import pickle
import os
from utils import get_sensor_data
# from hdbscan.hdbscan.hdbscan_ import HDBSCAN
# from hdbscan.hdbscan.prediction import approximate_predict

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
  clusterer = load_clusterer('clusterers/' + pair_dict[fieldname]
                         + '_p' + str(number) + '.pkl')
  pred_label, strength = approximate_predict(clusterer, [[data]])
  if pred_label == -1:
    return True
  return False




def approximate_predict(clusterer, points_to_predict):

    if clusterer.prediction_data_ is None:
        raise ValueError('Clusterer does not have prediction data!'
                         ' Try fitting with prediction_data=True set,'
                         ' or run generate_prediction_data on the clusterer')

    points_to_predict = np.asarray(points_to_predict)

    if points_to_predict.shape[1] != \
            clusterer.prediction_data_.raw_data.shape[1]:
        raise ValueError('New points dimension does not match fit data!')

    if clusterer.prediction_data_.cluster_tree.shape[0] == 0:
        warn('Clusterer does not have any defined clusters, new data'
             ' will be automatically predicted as noise.')
        labels = -1 * np.ones(points_to_predict.shape[0], dtype=np.int32)
        probabilities = np.zeros(points_to_predict.shape[0], dtype=np.float32)
        return labels, probabilities

    labels = np.empty(points_to_predict.shape[0], dtype=np.int)
    probabilities = np.empty(points_to_predict.shape[0], dtype=np.float64)

    min_samples = clusterer.min_samples or clusterer.min_cluster_size
    neighbor_distances, neighbor_indices = \
        clusterer.prediction_data_.tree.query(points_to_predict,
                                              k=2 * min_samples)

    for i in range(points_to_predict.shape[0]):
        label, prob = _find_cluster_and_probability(
            clusterer.condensed_tree_,
            clusterer.prediction_data_.cluster_tree,
            neighbor_indices[i],
            neighbor_distances[i],
            clusterer.prediction_data_.core_distances,
            clusterer.prediction_data_.cluster_map,
            clusterer.prediction_data_.max_lambdas,
            min_samples
        )
        labels[i] = label
        probabilities[i] = prob

    return labels, probabilities
