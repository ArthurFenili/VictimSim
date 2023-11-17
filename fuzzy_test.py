import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import os
import csv
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import KFold

class FuzzySystem:
    def __init__(self, lista_das_infos_das_vitimas):
        #print("lista_das_infos_das_vitimas: ", lista_das_infos_das_vitimas)
        self.number_of_victims = 0
        self.lista_das_infos_das_vitimas = lista_das_infos_das_vitimas
        self.resultados = []
        self.situacao_string = []
        self.gravity = []
        self.qpa_input = []
        self.pulso_input = []
        self.respiracao_input = []
        self.classe_gravidade = []
        for vitima_info in lista_das_infos_das_vitimas:
            self.number_of_victims += 1
            #print("vitima_info: ", vitima_info)
            self.qpa_input.append(float(vitima_info[3]))
            self.pulso_input.append(float(vitima_info[4]))
            self.respiracao_input.append(float(vitima_info[5]))
        self.define_linguistic_variables()
        self.define_pertinencias()


    def determine_status(self,pressure, pulse, respiration, simulator):
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
            self.situacao_string.append("Stable")
            self.resultados.append(4)
        elif gravity_status <=42.5: 
            self.situacao_string.append("Potentially Stable")
            self.resultados.append(3)
        elif gravity_status <= 84: 
            self.situacao_string.append("Unstable")
            self.resultados.append(2)
        else:
            self.situacao_string.append("Critical")
            self.resultados.append(1)

    def define_linguistic_variables(self):
    # Step 2: Define Linguistic Variables
        self.pressao = ctrl.Antecedent(np.arange(-10, 10, 0.1), 'qualidade_pressao')
        self.pulso = ctrl.Antecedent(np.arange(0, 200, 0.1), 'pulso')
        self.respiracao = ctrl.Antecedent(np.arange(0, 22, 0.1), 'respiracao')
        self.gravity = ctrl.Consequent(np.arange(0, 100, 0.1), 'gravity')

    def define_pertinencias(self):
    # Step 3: Specify Membership Functions
    #qualidade da pressão, onde 0 é a qualidade máxima -10 é a pior
    #qualidade quando a pressão está excessivamente baixa, +10 é a pior qualidade quando
    #a pressão está excessivamente alta
    # Funções de pertinência para Pressão
        self.pressao['low'] = fuzz.trimf(self.pressao.universe, [-10, -4, 0])
        self.pressao['good'] = fuzz.trimf(self.pressao.universe, [-3, 0, 5])
        self.pressao['high'] = fuzz.trimf(self.pressao.universe, [2, 8, 10])

        # Funções de pertinência para Pulso
        self.pulso['low'] = fuzz.trimf(self.pulso.universe, [0, 35, 51])
        self.pulso['medium'] = fuzz.trimf(self.pulso.universe, [45, 70, 130])
        self.pulso['high'] = fuzz.trimf(self.pulso.universe, [120, 141, 200])

        # Funções de pertinência para Respiração
        self.respiracao['low'] = fuzz.trimf(self.respiracao.universe, [0, 6, 14])
        self.respiracao['medium'] = fuzz.trimf(self.respiracao.universe, [8, 15, 21])
        self.respiracao['high'] = fuzz.trimf(self.respiracao.universe, [17, 21, 22])

        # Funções de pertinência para Gravidade
        self.gravity['low'] = fuzz.trimf(self.gravity.universe, [0, 15, 25])
        self.gravity['medium'] = fuzz.trimf(self.gravity.universe, [20, 32, 47])
        self.gravity['high'] = fuzz.trimf(self.gravity.universe, [40, 52, 80]) 
        self.gravity['very_high'] = fuzz.trimf(self.gravity.universe, [70, 90, 100])

    def define_rules_and_infere_system(self):
        pressao = self.pressao
        pulso = self.pulso
        respiracao = self.respiracao
        gravity = self.gravity
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

        for i in range(self.number_of_victims):
            pressure_value = self.qpa_input[i]
            pulse_value = self.pulso_input[i]
            respiration_value = self.respiracao_input[i]
            self.determine_status(pressure_value, pulse_value, respiration_value, simulator)
            
        return self.resultados