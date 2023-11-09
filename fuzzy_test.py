import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import os
import csv
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import KFold


def determine_status(pressure, pulse, respiration):
    # Input values
    simulator.input['qualidade_pressao'] = pressure
    simulator.input['pulso'] = pulse
    simulator.input['respiracao'] = respiration

    # Perform the fuzzy inference
    simulator.compute()

    # Get the output value
    gravity_status = simulator.output['gravity']
    #print("simulator.output['gravity']: ", simulator.output['gravity'])


    # Determine the situation based on the gravity_status
    if gravity_status <= 18: 
        situacao.append("Stable")
        gravidade.append(4)
    elif gravity_status <=42.5: 
        situacao.append("Potentially Stable")
        gravidade.append(3)
    elif gravity_status <= 84: 
        situacao.append("Unstable")
        gravidade.append(2)
    else:
        situacao.append("Critical")
        gravidade.append(1)

    
    # You can print or return the status if needed
    #print(f"Pressure: {pressure}, Pulse: {pulse}, Respiration: {respiration}, Status: {situacao[-1]}, Gravidade: {gravidade[-1]}")

id_input = []
qpa_input = []
pulso_input = []
respiracao_input = []
gravity_output = []
situacao = []
classe_gravidade = []
gravidade = []

# Step 1: Read Data
vs_file = os.path.join("datasets", "data_800vic", "sinais_vitais_com_label.txt" )
number_of_victims = 0
with open(vs_file, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        number_of_victims += 1
        id_input.append(float(row[0]))
        qpa_input.append(float(row[1]))
        pulso_input.append(float(row[2]))
        respiracao_input.append(float(row[3]))
        

# Step 2: Define Linguistic Variables
pressao = ctrl.Antecedent(np.arange(-10, 10, 0.1), 'qualidade_pressao')
pulso = ctrl.Antecedent(np.arange(0, 200, 0.1), 'pulso')
respiracao = ctrl.Antecedent(np.arange(0, 22, 0.1), 'respiracao')
gravity = ctrl.Consequent(np.arange(0, 100, 0.1), 'gravity')

# Step 3: Specify Membership Functions
#qualidade da pressão, onde 0 é a qualidade máxima -10 é a pior
#qualidade quando a pressão está excessivamente baixa, +10 é a pior qualidade quando
#a pressão está excessivamente alta
 # Funções de pertinência para Pressão
pressao['low'] = fuzz.trimf(pressao.universe, [-10, -4, 0])
pressao['good'] = fuzz.trimf(pressao.universe, [-3, 0, 5])
pressao['high'] = fuzz.trimf(pressao.universe, [2, 8, 10])

# Funções de pertinência para Pulso
pulso['low'] = fuzz.trimf(pulso.universe, [0, 35, 51])
pulso['medium'] = fuzz.trimf(pulso.universe, [45, 70, 130])
pulso['high'] = fuzz.trimf(pulso.universe, [120, 141, 200])

# Funções de pertinência para Respiração
respiracao['low'] = fuzz.trimf(respiracao.universe, [0, 6, 14])
respiracao['medium'] = fuzz.trimf(respiracao.universe, [8, 15, 21])
respiracao['high'] = fuzz.trimf(respiracao.universe, [17, 21, 22])

# Funções de pertinência para Gravidade
gravity['low'] = fuzz.trimf(gravity.universe, [0, 15, 25])
gravity['medium'] = fuzz.trimf(gravity.universe, [20, 32, 47])
gravity['high'] = fuzz.trimf(gravity.universe, [40, 52, 80]) 
gravity['very_high'] = fuzz.trimf(gravity.universe, [70, 90, 100])

# Step 4: Define Fuzzy Rules
rule1 = ctrl.Rule(pressao['low'] & pulso['low'] & respiracao['low'], gravity['high']) #
rule2 = ctrl.Rule(pressao['low'] & pulso['low'] & respiracao['medium'], gravity['medium'])
rule3 = ctrl.Rule(pressao['low'] & pulso['low'] & respiracao['high'], gravity['medium'])

rule4 = ctrl.Rule(pressao['low'] & pulso['medium'] & respiracao['low'], gravity['low'])
rule5 = ctrl.Rule(pressao['low'] & pulso['medium'] & respiracao['medium'], gravity['low'])
rule6 = ctrl.Rule(pressao['low'] & pulso['medium'] & respiracao['high'], gravity['medium'])

rule7 = ctrl.Rule(pressao['low'] & pulso['high'] & respiracao['low'], gravity['high']) #
rule8 = ctrl.Rule(pressao['low'] & pulso['high'] & respiracao['medium'], gravity['medium'])
rule9 = ctrl.Rule(pressao['low'] & pulso['high'] & respiracao['high'], gravity['very_high']) #

rule10 = ctrl.Rule(pressao['good'] & pulso['low'] & respiracao['low'], gravity['high'])
rule11 = ctrl.Rule(pressao['good'] & pulso['low'] & respiracao['medium'], gravity['low']) #
rule12 = ctrl.Rule(pressao['good'] & pulso['low'] & respiracao['high'], gravity['medium']) #

rule13 = ctrl.Rule(pressao['good'] & pulso['medium'] & respiracao['low'], gravity['low']) #
rule14 = ctrl.Rule(pressao['good'] & pulso['medium'] & respiracao['medium'], gravity['low']) #
rule15 = ctrl.Rule(pressao['good'] & pulso['medium'] & respiracao['high'], gravity['low'])

rule16 = ctrl.Rule(pressao['good'] & pulso['high'] & respiracao['low'], gravity['medium'])
rule17 = ctrl.Rule(pressao['good'] & pulso['high'] & respiracao['medium'], gravity['high'])
rule18 = ctrl.Rule(pressao['good'] & pulso['high'] & respiracao['high'], gravity['high'])

rule19 = ctrl.Rule(pressao['high'] & pulso['low'] & respiracao['low'], gravity['high'])
rule20 = ctrl.Rule(pressao['high'] & pulso['low'] & respiracao['medium'], gravity['high']) #
rule21 = ctrl.Rule(pressao['high'] & pulso['low'] & respiracao['high'], gravity['very_high']) #

rule22 = ctrl.Rule(pressao['high'] & pulso['medium'] & respiracao['low'], gravity['high'])
rule23 = ctrl.Rule(pressao['high'] & pulso['medium'] & respiracao['medium'], gravity['medium'])
rule24 = ctrl.Rule(pressao['high'] & pulso['medium'] & respiracao['high'], gravity['very_high'])

rule25 = ctrl.Rule(pressao['high'] & pulso['high'] & respiracao['low'], gravity['very_high']) #
rule26 = ctrl.Rule(pressao['high'] & pulso['high'] & respiracao['medium'], gravity['very_high']) #
rule27 = ctrl.Rule(pressao['high'] & pulso['high'] & respiracao['high'], gravity['very_high']) #

# Step 5: Fuzzy Inference (using rule aggregation method)
system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27])
simulator = ctrl.ControlSystemSimulation(system)

for i in range(number_of_victims):
    #print(qpa_input[i], pulso_input[i], respiracao_input[i])
    pressure_value = qpa_input[i]
    pulse_value = pulso_input[i]
    respiration_value = respiracao_input[i]
    determine_status(pressure_value, pulse_value, respiration_value)
    
    
precision = precision_score(classe_gravidade, gravidade, average='weighted')
recall = recall_score(classe_gravidade, gravidade, average='weighted')
f1 = f1_score(classe_gravidade, gravidade, average='weighted')
accuracy = accuracy_score(classe_gravidade, gravidade)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-measure: {f1}")
print(f"Acurácia: {accuracy}")

num_folds = 5
kf = KFold(n_splits=num_folds)

# Dados de entrada
X = np.column_stack((qpa_input, pulso_input, respiracao_input))

# Classes reais
y = classe_gravidade

# Inicialize listas para armazenar métricas de desempenho
precisions = []
recalls = []
f1_scores = []
accuracies = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

    # Crie um novo sistema de controle para cada fold (se necessário)
    system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27])
    simulator = ctrl.ControlSystemSimulation(system)

    # Realize a classificação para o fold atual
    predicted_gravity = []
    for i in range(len(X_test)):
        pressure_value, pulse_value, respiration_value = X_test[i]
        determine_status(pressure_value, pulse_value, respiration_value)
        predicted_gravity.append(gravidade[-1])

    # Calcule as métricas para este fold
    precision = precision_score(y_test, predicted_gravity, average='weighted')
    recall = recall_score(y_test, predicted_gravity, average='weighted')
    f1 = f1_score(y_test, predicted_gravity, average='weighted')
    accuracy = accuracy_score(y_test, predicted_gravity)

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    accuracies.append(accuracy)

# Calcule a média das métricas de desempenho em todos os folds
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)
mean_f1 = np.mean(f1_scores)
mean_accuracy = np.mean(accuracies)

print(f"Mean Precision: {mean_precision}")
print(f"Mean Recall: {mean_recall}")
print(f"Mean F-measure: {mean_f1}")
print(f"Mean Accuracy: {mean_accuracy}")