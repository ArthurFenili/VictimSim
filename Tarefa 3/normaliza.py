import pandas as pd
import numpy as np

old_min = -10
old_max = 10
new_min = 0
new_max = 100
# Carregar o arquivo de texto (substitua 'seu_arquivo.txt' pelo nome do seu arquivo)
arquivo_path = 'sinais_vitais_teste1.txt'
df = pd.read_csv(arquivo_path, sep=',', header=None)  # Dependendo do formato, ajuste o separador (sep)
print(df)
# Normalizar os dados da coluna 0
for index, row in df.iterrows():
    row[0] = np.round(np.interp(row[0], (old_min, old_max), (new_min, new_max)), 3)

print(df)
# Salvar de volta no arquivo de texto
df.to_csv('sinais_teste_normalizados.txt', sep=',', index=False)  # Dependendo do formato, ajuste o separador (sep)
