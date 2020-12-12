from fastapi import FastAPI
import os
# from typing import Optional
from utils import get_weekly_field, get_sensor_data, get_dates_from_series
from ml.conditions import usage

app = FastAPI()

SENSOR_PATH = os.path.join('data', 'sensor_data.csv')
SENSOR_DATA = get_sensor_data(SENSOR_PATH)

@app.get('/greenhouse/{grh_num}')
def get_greenhouse_plot_data(grh_num=1):
    air_temperatures = get_weekly_field(SENSOR_DATA, 'air_temperature')
    return {
        'temperatures': list(air_temperatures.values),
        'lightningLevels': list(get_weekly_field(SENSOR_DATA, 'illumination_p' + str(grh_num)).values),
        'soilMoisture': list(get_weekly_field(SENSOR_DATA, 'relative_soil_mosture_p' + str(grh_num)).values),
        'soilAcidity': list(get_weekly_field(SENSOR_DATA, 'relative_soil_mosture_p' + str(grh_num)).values),
        'date': list(get_dates_from_series(air_temperatures))
    }

@app.get('/getoptimaldata')
def get_optimal_data():
    return usage.optimal_values()