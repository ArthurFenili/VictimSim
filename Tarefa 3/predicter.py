
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from joblib import dump, load
from sklearn.preprocessing import StandardScaler

#read a txt file with 2 columns of features
#load the model

best_model_file = 'model2.joblib'
best_model = load(best_model_file)

#read the test data
test_data_file = 'sinais_teste_normalizados.txt'
test_data = pd.read_csv(test_data_file, sep=',', header=None)

#select columns 1 2 3 from test_Data
test_data = test_data.drop(index=0, errors='ignore')
test_data = test_data.iloc[:, [0,1,2]]
print(test_data)


predictions = best_model.predict(test_data)
resultado = []
print(predictions)
# for prediction in predictions:
#     value = float(prediction[0])
#     resultado.append(value/10)

# print(resultado)
# resultado_df = pd.DataFrame(resultado)

# resultado_df.to_csv('resultado.txt', sep=',', index=False)  # Dependendo do formato, ajuste o separador (sep)