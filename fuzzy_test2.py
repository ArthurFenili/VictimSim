# Importar as bibliotecas necessárias
import os
import csv
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from skfuzzy import control as ctrl
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Criar listas vazias para armazenar os valores das variáveis de entrada e saída
id_input = []
qpa_input = []
pulso_input = []
respiracao_input = []
classe_gravidade = []

# Ler os dados do arquivo sinais_vitais.txt
vs_file = os.path.join("datasets", "data_12x12_10vic", "sinais_vitais.txt" )
number_of_victims = 0 # Contador para o número de vítimas
with open(vs_file, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        number_of_victims += 1 # Incrementar o contador
        id_input.append(float(row[0])) # Adicionar o valor da coluna 0 (id) à lista id_input
        qpa_input.append(float(row[3])) # Adicionar o valor da coluna 3 (qualidade da pressão arterial) à lista qpa_input
        pulso_input.append(float(row[4])) # Adicionar o valor da coluna 4 (pulso) à lista pulso_input
        respiracao_input.append(float(row[5])) # Adicionar o valor da coluna 5 (respiração) à lista respiracao_input
        classe_gravidade.append(float(row[7])) # Adicionar o valor da coluna 7 (classe de gravidade) à lista classe_gravidade

# Definir as variáveis linguísticas e os seus respectivos universos
pressao = ctrl.Antecedent(np.arange(-10, 10, 0.1), 'qualidade_pressao')
pulso = ctrl.Antecedent(np.arange(0, 200, 0.1), 'pulso')
respiracao = ctrl.Antecedent(np.arange(0, 22, 0.1), 'respiracao')
gravity = ctrl.Consequent(np.arange(0, 100, 0.1), 'gravity')

# Especificar as funções de pertinência para cada variável linguística
# Usando a função trimf da biblioteca skfuzzy para criar conjuntos fuzzy triangulares
# Com os mesmos nomes e parâmetros que você me deu

# Funções de pertinência para Pressão
pressao['verylow'] = fuzz.trimf(pressao.universe, [-10, -7, -5])
pressao['low'] = fuzz.trimf(pressao.universe, [-7, -3, 0])
pressao['medium'] = fuzz.trimf(pressao.universe, [-3, 0, 4])
pressao['high'] = fuzz.trimf(pressao.universe, [1, 5, 7])
pressao['veryhigh'] = fuzz.trimf(pressao.universe, [6, 9, 10])

# Funções de pertinência para Pulso
pulso['verylow'] = fuzz.trimf(pulso.universe, [0, 25, 35])
pulso['low'] = fuzz.trimf(pulso.universe, [30, 45, 61])
pulso['medium'] = fuzz.trimf(pulso.universe, [45, 70, 130])
pulso['high'] = fuzz.trimf(pulso.universe, [120, 130, 180])
pulso['veryhigh'] = fuzz.trimf(pulso.universe, [170, 190, 200])

# Funções de pertinência para Respiração
respiracao['verylow'] = fuzz.trimf(respiracao.universe, [0, 4, 10])
respiracao['low'] = fuzz.trimf(respiracao.universe, [4, 8, 14])
respiracao['medium'] = fuzz.trimf(respiracao.universe, [9, 15, 19])
respiracao['high'] = fuzz.trimf(respiracao.universe, [16, 18, 21])
respiracao['veryhigh'] = fuzz.trimf(respiracao.universe, [18, 21, 22])

# Funções de pertinência para Gravidade
gravity['low'] = fuzz.trimf(gravity.universe, [0, 15, 25])
gravity['medium'] = fuzz.trimf(gravity.universe, [20, 32, 47])
gravity['high'] = fuzz.trimf(gravity.universe, [40, 52, 80]) 
gravity['very high'] = fuzz.trimf(gravity.universe, [70, 90, 100])

# Gerar as regras fuzzy usando o algoritmo de Wang-Mendel
regras = {} # Dicionário para armazenar as regras
for i in range(number_of_victims): # Para cada vítima no conjunto de dados
    # Fuzzificar os dados de entrada
    pressao_valor = qpa_input[i] # Valor da qualidade da pressão arterial
    pulso_valor = pulso_input[i] # Valor do pulso
    respiracao_valor = respiracao_input[i] # Valor da respiração

    grau_pressao_verylow = fuzz.interp_membership(pressao.universe, pressao['verylow'].mf, pressao_valor) # Grau de pertinência da qualidade da pressão arterial no conjunto baixo
    grau_pressao_low = fuzz.interp_membership(pressao.universe, pressao['low'].mf, pressao_valor) # Grau de pertinência da qualidade da pressão arterial no conjunto baixo
    grau_pressao_medium = fuzz.interp_membership(pressao.universe, pressao['medium'].mf, pressao_valor) # Grau de pertinência da qualidade da pressão arterial no conjunto bom
    grau_pressao_high = fuzz.interp_membership(pressao.universe, pressao['high'].mf, pressao_valor) # Grau de pertinência da qualidade da pressão arterial no conjunto alto
    grau_pressao_veryhigh = fuzz.interp_membership(pressao.universe, pressao['veryhigh'].mf, pressao_valor) # Grau de pertinência da qualidade da pressão arterial no conjunto baixo
    grau_pulso_verylow = fuzz.interp_membership(pulso.universe, pulso['verylow'].mf, pulso_valor) # Grau de pertinência do pulso no conjunto baixo
    grau_pulso_low = fuzz.interp_membership(pulso.universe, pulso['low'].mf, pulso_valor) # Grau de pertinência do pulso no conjunto baixo
    grau_pulso_medium = fuzz.interp_membership(pulso.universe, pulso['medium'].mf, pulso_valor) # Grau de pertinência do pulso no conjunto médio
    grau_pulso_high = fuzz.interp_membership(pulso.universe, pulso['high'].mf, pulso_valor) # Grau de pertinência do pulso no conjunto alto
    grau_pulso_veryhigh = fuzz.interp_membership(pulso.universe, pulso['veryhigh'].mf, pulso_valor) # Grau de pertinência do pulso no conjunto baixo
    grau_respiracao_verylow = fuzz.interp_membership(respiracao.universe, respiracao['verylow'].mf, respiracao_valor) # Grau de pertinência da respiração no conjunto baixo
    grau_respiracao_low = fuzz.interp_membership(respiracao.universe, respiracao['low'].mf, respiracao_valor) # Grau de pertinência da respiração no conjunto baixo
    grau_respiracao_medium = fuzz.interp_membership(respiracao.universe, respiracao['medium'].mf, respiracao_valor) # Grau de pertinência da respiração no conjunto médio
    grau_respiracao_high = fuzz.interp_membership(respiracao.universe, respiracao['high'].mf, respiracao_valor) # Grau de pertinência da respiração no conjunto alto
    grau_respiracao_veryhigh = fuzz.interp_membership(respiracao.universe, respiracao['veryhigh'].mf, respiracao_valor) # Grau de pertinência da respiração no conjunto baixo
    
    # Aplicar o operador mínimo para obter o grau de confiança de cada regra possível
    # Considerando todas as combinações de conjuntos fuzzy das variáveis de entrada e saída
    # Usando uma lista para armazenar os graus de confiança das 81 regras possíveis
    graus_regras = []
    for pressao_conjunto in pressao.terms: # Para cada conjunto fuzzy da variável pressao
        for pulso_conjunto in pulso.terms: # Para cada conjunto fuzzy da variável pulso
            for respiracao_conjunto in respiracao.terms: # Para cada conjunto fuzzy da variável respiracao
                for gravity_conjunto in gravity.terms: # Para cada conjunto fuzzy da variável de saída gravity
                    # Obter o grau de pertinência das variáveis de entrada nos respectivos conjuntos fuzzy
                    grau_pressao = eval(f"grau_pressao_{pressao_conjunto}")
                    grau_pulso = eval(f"grau_pulso_{pulso_conjunto}")
                    grau_respiracao = eval(f"grau_respiracao_{respiracao_conjunto}")
                    # Aplicar o operador mínimo entre os graus de pertinência das variáveis de entrada
                    # Obtendo o grau de confiança da regra
                    #grau_regra = np.median(grau_pressao, grau_pulso, grau_respiracao)
                    values = np.array([grau_pressao, grau_pulso, grau_respiracao])
                    grau_regra = np.mean(values)
                    # Adicionar o grau de confiança da regra à lista graus_regras
                    graus_regras.append(grau_regra)

regras = {} # Dicionário para armazenar as regras
indice = 0 # Índice para percorrer a lista graus_regras
for pressao_conjunto in pressao.terms: # Para cada conjunto fuzzy da variável pressao
    for pulso_conjunto in pulso.terms: # Para cada conjunto fuzzy da variável pulso
        for respiracao_conjunto in respiracao.terms: # Para cada conjunto fuzzy da variável respiracao
            for gravity_conjunto in gravity.terms: # Para cada conjunto fuzzy da variável de saída gravity
                # Obter o grau de confiança da regra na lista graus_regras
                grau_regra = graus_regras[indice]
                # Incrementar o índice
                indice += 1
                if grau_regra > 0: # Se a regra tem algum grau de confiança
                    # Criar a chave do dicionário com a combinação dos conjuntos fuzzy das variáveis de entrada
                    chave = f"{pressao_conjunto}_{pulso_conjunto}_{respiracao_conjunto}"
                    # Verificar se a chave já existe no dicionário
                    if chave in regras:
                        # Se existe, comparar o grau de confiança da regra atual com o da regra armazenada
                        if grau_regra > regras[chave][1]:
                            # Se o grau de confiança da regra atual for maior, substituir a regra armazenada pela regra atual
                            regras[chave] = (gravity_conjunto, grau_regra)
                    else:
                        # Se a chave não existe no dicionário, adicionar a regra ao dicionário
                        regras[chave] = (gravity_conjunto, grau_regra)

# Criar uma lista vazia para armazenar as regras do sistema de controle fuzzy
regras_ctrl = []

# Para cada chave (combinação de conjuntos fuzzy das variáveis de entrada) no dicionário de regras
for chave in regras:
    # Obter o conjunto fuzzy da variável de saída e o grau de confiança da regra
    gravity_conjunto, grau_regra = regras[chave]
    # Obter os conjuntos fuzzy das variáveis de entrada da chave
    pressao_conjunto, pulso_conjunto, respiracao_conjunto = chave.split("_")
    # Criar a regra fuzzy
    regra = ctrl.Rule(pressao[pressao_conjunto] & pulso[pulso_conjunto] & respiracao[respiracao_conjunto], gravity[gravity_conjunto])
    # Adicionar a regra à lista de regras do sistema de controle fuzzy
    regras_ctrl.append(regra)

# Criar o sistema de controle fuzzy com as regras

sistema_ctrl = ctrl.ControlSystem(regras_ctrl)

# Criar a simulação do sistema de controle fuzzy
sistema = ctrl.ControlSystemSimulation(sistema_ctrl)

#-------------------------------------------------------------

split_index = int(len(id_input) * 0.8)
id_input_treino, qpa_input_treino, pulso_input_treino, respiracao_input_treino, classe_gravidade_treino = (
    id_input[:split_index],
    qpa_input[:split_index],
    pulso_input[:split_index],
    respiracao_input[:split_index],
    classe_gravidade[:split_index],
)

id_input_teste, qpa_input_teste, pulso_input_teste, respiracao_input_teste, classe_gravidade_teste = (
    id_input[split_index:],
    qpa_input[split_index:],
    pulso_input[split_index:],
    respiracao_input[split_index:],
    classe_gravidade[split_index:],
)

for i in range(len(id_input_treino)):
    sistema.input['qualidade_pressao'] = qpa_input_treino[i]
    sistema.input['pulso'] = pulso_input_treino[i]
    sistema.input['respiracao'] = respiracao_input_treino[i]
    sistema.compute()

predicted_classes = []
actual_classes = []

# Suponha que seus dados estejam em uma lista chamada 'data'
for i in range(len(id_input_treino)):
    sistema.input['qualidade_pressao'] = qpa_input_treino[i]
    sistema.input['pulso'] = pulso_input_treino[i]
    sistema.input['respiracao'] = respiracao_input_treino[i]
    sistema.compute()

    # Adicione a classe prevista e a real às listas
    predicted_classes.append(sistema.output['gravity'])
    actual_classes.append(classe_gravidade_treino[i])


predicted_labels_discrete = []
for pred in predicted_classes:
    if pred < 18:
        predicted_labels_discrete.append(4)
    elif pred < 42:
        predicted_labels_discrete.append(3)
    elif pred < 84:
        predicted_labels_discrete.append(2)
    else:
        predicted_labels_discrete.append(1)
    print ("Gravidade: ",pred)

print(predicted_labels_discrete)
print(actual_classes)

predicted_labels_discrete = [round(x) for x in predicted_labels_discrete]

# Agora que você tem todas as classes previstas e reais, você pode calcular as métricas
precision = precision_score(actual_classes, predicted_labels_discrete, average='macro')
recall = recall_score(actual_classes, predicted_labels_discrete, average='macro')
f_measure = f1_score(actual_classes, predicted_labels_discrete, average='macro')
accuracy = accuracy_score(actual_classes, predicted_labels_discrete)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F-Measure: {f_measure}')
print(f'Accuracy: {accuracy}')

#split_ratio = 0.8
#split_index = int(number_of_victims * split_ratio)
#
## Dados de treinamento
#train_pressao = qpa_input[:split_index]
#train_pulso = pulso_input[:split_index]
#train_respiracao = respiracao_input[:split_index]
#train_gravity = classe_gravidade[:split_index]
#
## Dados de validação
#val_pressao = qpa_input[split_index:]
#val_pulso = pulso_input[split_index:]
#val_respiracao = respiracao_input[split_index:]
#val_gravity = classe_gravidade[split_index:]
#
#for i in range(split_index):
#    # Fuzzificar os dados de treinamento
#    sistema.input['qualidade_pressao'] = train_pressao[i]
#    sistema.input['pulso'] = train_pulso[i]
#    sistema.input['respiracao'] = train_respiracao[i]
#
#    # Computar a saída do sistema
#    sistema.compute()
#
#    # Atualizar os parâmetros do sistema de controle fuzzy usando a saída desejada
#    sistema.output['gravity'] = train_gravity[i]
#
#    #sistema_ctrl.compute()
#
## Classificação e Avaliação
#predicted_labels = []
#true_labels = []
#
#for i in range(split_index, number_of_victims):
#    # Fuzzificar os dados de validação
#    sistema.input['qualidade_pressao'] = val_pressao[i - split_index]
#    sistema.input['pulso'] = val_pulso[i - split_index]
#    sistema.input['respiracao'] = val_respiracao[i - split_index]
#
#    # Computar a saída do sistema
#    sistema.compute()
#
#    # Obter a classe prevista
#    predicted_class = sistema.output['gravity']
#    predicted_labels.append(predicted_class)
#
#    # Obter a classe real
#    true_class = val_gravity[i - split_index]
#    true_labels.append(true_class)
#
#
#limiar_low = 25
#limiar_medium = 50
#limiar_high = 75
#
## Converta as previsões contínuas em rótulos de classe
#predicted_labels_discrete = []
#
#for pred in predicted_labels:
#    if pred < limiar_low:
#        predicted_labels_discrete.append(4)
#    elif pred < limiar_medium:
#        predicted_labels_discrete.append(3)
#    elif pred < limiar_high:
#        predicted_labels_discrete.append(2)
#    else:
#        predicted_labels_discrete.append(1)
#    print ("Gravidade: ",pred)
#
## Calcular o F-measure
#f_measure = f1_score(true_labels, predicted_labels_discrete, average='weighted')
#print(f'F-measure: {f_measure}')