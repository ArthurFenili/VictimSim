import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import os
import csv

def determine_status(pressure_value, pulse_value, respiration_value):
    # Set input values for the simulation
    simulator.input['qualidade_pressao'] = pressure_value
    simulator.input['pulso'] = pulse_value
    simulator.input['respiracao'] = respiration_value

    # Compute the output (status) using the fuzzy logic rules
    simulator.compute()

    # Retrieve the determined status (you may need to adjust this based on your fuzzy sets)
    status_label = fuzz.interp_membership(gravity.universe, simulator.output['gravity'], 0.5)

    print("Status: ", status_label)

id_input = []
qpa_input = []
pulso_input = []
respiracao_input = []
gravity_output = []
situacao = []

# Step 1: Read Data
vs_file = os.path.join("datasets", "data_12x12_10vic", "sinais_teste_fuzzy.txt" )

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
gravity = ctrl.Consequent(np.arange(0, 50, 0.1), 'gravity')

# Step 3: Specify Membership Functions
#qualidade da pressão, onde 0 é a qualidade máxima -10 é a pior
#qualidade quando a pressão está excessivamente baixa, +10 é a pior qualidade quando
#a pressão está excessivamente alta
pressao['very_low'] = fuzz.trimf(pressao.universe, [-10, -10, -5])
pressao['low'] = fuzz.trimf(pressao.universe, [-7, -5, -2])
pressao['good'] = fuzz.trimf(pressao.universe, [-5, 0, 5])
pressao['high'] = fuzz.trimf(pressao.universe, [2, 5, 7])
pressao['very_high'] = fuzz.trimf(pressao.universe, [5, 10, 10])

#pulso, onde o pulso de uma pessoa normal, em repouso, é de 60 a 100 batimentos por minuto
pulso['very_low'] = fuzz.trimf(pulso.universe, [0, 0, 30])
pulso['low'] = fuzz.trimf(pulso.universe, [0, 40, 80])
pulso['good'] = fuzz.trimf(pulso.universe, [50, 80, 130])
pulso['high'] = fuzz.trimf(pulso.universe, [110, 160, 180])
pulso['very_high'] = fuzz.trimf(pulso.universe, [150, 200, 200])

#respiração, onde a respiração normal é de 12 a 20 respirações por minuto
respiracao['very_low'] = fuzz.trimf(respiracao.universe, [0, 0, 5])
respiracao['low'] = fuzz.trimf(respiracao.universe, [0, 5, 10])
respiracao['good'] = fuzz.trimf(respiracao.universe, [8, 15, 22])
respiracao['high'] = fuzz.trimf(respiracao.universe, [20, 22, 22])

#gravidade, onde 0 é a gravidade mínima e 100 é a gravidade máxima
gravity['stable'] = fuzz.trimf(gravity.universe, [0, 0, 25])
gravity['potentially_stable'] = fuzz.trimf(gravity.universe, [0, 25, 50])
gravity['unstable'] = fuzz.trimf(gravity.universe, [25, 50, 75])
gravity['critic'] = fuzz.trimf(gravity.universe, [50, 100, 100])

# Step 4: Define Fuzzy Rules
rule0 = ctrl.Rule(pressao['low'] & pulso['low'] & respiracao['very_low'], gravity['critic'])
rule1 = ctrl.Rule(pressao['very_low'] & pulso['very_low'] & respiracao['very_low'], gravity['critic']) 
rule2 = ctrl.Rule(pressao['very_low'] & pulso['very_low'] & respiracao['low'], gravity['critic']) 
rule3 = ctrl.Rule(pressao['very_high'] & pulso['very_low'] & respiracao['very_low'], gravity['critic']) 
rule4 = ctrl.Rule(pressao['very_high'] & pulso['very_low'] & respiracao['low'], gravity['critic']) 

rule5 = ctrl.Rule(pressao['low'] & pulso['low'] & respiracao['low'], gravity['unstable'])
rule6 = ctrl.Rule(pressao['high'] & pulso['high'] & respiracao['low'], gravity['unstable'])
rule7 = ctrl.Rule(pressao['low'] & pulso['high'] & respiracao['low'], gravity['unstable'])
rule8 = ctrl.Rule(pressao['high'] & pulso['low'] & respiracao['low'], gravity['unstable'])

rule9 = ctrl.Rule(pressao['low'] & pulso['low'] & respiracao['good'], gravity['potentially_stable'])
rule10 = ctrl.Rule(pressao['high'] & pulso['high'] & respiracao['good'], gravity['potentially_stable'])
rule11 = ctrl.Rule(pressao['low'] & pulso['high'] & respiracao['good'], gravity['potentially_stable'])
rule12 = ctrl.Rule(pressao['high'] & pulso['low'] & respiracao['good'], gravity['potentially_stable'])

rule13 = ctrl.Rule(pressao['good'] & pulso['low'] & respiracao['good'], gravity['stable'])
rule14 = ctrl.Rule(pressao['low'] & pulso['good'] & respiracao['good'], gravity['stable'])
rule15 = ctrl.Rule(pressao['good'] & pulso['good'] & respiracao['good'], gravity['stable'])
rule16 = ctrl.Rule(pressao['good'] & pulso['good'] & respiracao['high'], gravity['stable'])

# Step 5: Fuzzy Inference (using rule aggregation method)
system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16])
simulator = ctrl.ControlSystemSimulation(system)

for i in range(number_of_victims):
    print(qpa_input[i], pulso_input[i], respiracao_input[i])
    pressure_value = qpa_input[i]
    pulse_value = pulso_input[i]
    respiration_value = respiracao_input[i]
    determine_status(pressure_value, pulse_value, respiration_value)
    
    


