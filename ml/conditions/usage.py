import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import json

sensors = pd.read_csv('../../data/sensor_data.csv')

sensors.index = sensors.index[::-1]
sensors.sort_index(inplace=True)

model = CatBoostRegressor()
model = model.load_model("predictions")

#sensors

# fake data just for debugging
p1 = sensors.iloc[:,[2,3,4,5]][405:406]
p2 = sensors.iloc[:,[2,6,7,8]][306:307]
p3 = sensors.iloc[:,[2,9,10,11]][546:547]
p4 = sensors.iloc[:,[2,12,13,14]][783:784]
p5 = sensors.iloc[:,[2,15,16,17]][1758:1759]

def ranked_preds(d):
    data = d
    t_range = np.linspace(sensors.air_temperature.min(), sensors.air_temperature.max(), num = 10)
    m_range = np.linspace(sensors.relative_soil_mosture_p1.min(), sensors.relative_soil_mosture_p1.max(), num = 10)
    il_range = np.linspace(sensors.illumination_p1.min(), sensors.illumination_p1.max(), num = 10)
    key = np.array([1]*10)

    temp = pd.DataFrame({'key':key, 't_range':t_range})
    moist = pd.DataFrame({'key':key, 'm_range':m_range})
    ill = pd.DataFrame({'key':key, 'il_range':il_range})

    # Castesian product 
    df = pd.merge(temp, moist, on='key')
    df = pd.merge(df, ill, on='key')

    data['key'] = 1
    p = data.iloc[:,[3,4]]
    predicted_params = pd.merge(df, p, on='key')
        
    predicted_params.drop(columns='key', inplace=True)
    #predicted_params = predicted_params.iloc[:,[3,0,1,2,4]]

    cols = ['air_temperature','relative_soil_mosture_p1','illumination_p1','soil_acidity_p1']
    # renaming for model prediction
    predicted_params.columns = cols

    preds = model.predict(predicted_params)
    return predicted_params[preds.argmax():preds.argmax()+1]

def get_optimal(data):
    rp = ranked_preds(data)
    data = data.drop(columns='key')
    data = data.fillna(0)

    before = data.to_numpy()
    after = rp.to_numpy()
    optimal = after - before
    return optimal

def optimal_values():
    g = globals()
    dict_to_js = {'data':[]
         }
        
    labels = ['currentTemperature:', 'currentLightingLevel:', 
        'currentSoilMoisture:', 'currentSoilAcidity:', 
        'optimalTemperature:', 'optimalLightingLevel:', 
        'optimalSoilMoisture:', 'optimalSoilAcidity:'
    ]
    
    for i in range(1,5):
        g['g{0}'.format(i)] = get_optimal(g['p{0}'.format(i)])
        g['ex_{0}'.format(i)] = g['p{0}'.format(i)].drop(columns='key').values.tolist()[0] + list(g['g{0}'.format(i)][0])

        d = dict(zip(labels,g['ex_{0}'.format(i)]))
        dict_to_js['data'].append(d) 
        
    return dict_to_js

