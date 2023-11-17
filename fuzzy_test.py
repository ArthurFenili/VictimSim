import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import os
import csv

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import os
import csv
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import KFold

class Fuzzy:
    def __init__(self):
        self.simulator = []
        self.id_input = []
        self.qpa_input = []
        self.pulso_input = []
        self.respiracao_input = []
        self.gravity_output = []
        self.situacao = []
        self.classe_gravidade = []
        self.gravidade = []
        self.new_gravity = "new_gravity.csv"

    def determine_status(self, pressure, pulse, respiration, id):
        # Input values
        self.simulator.input['qualidade_pressao'] = pressure
        self.simulator.input['pulso'] = pulse
        self.simulator.input['respiracao'] = respiration

        # Perform the fuzzy inference
        self.simulator.compute()

        # Get the output value
        gravity_status = self.simulator.output['gravity']
        #print("simulator.output['gravity']: ", simulator.output['gravity'])


        # Determine the situation based on the gravity_status
        if gravity_status <= 18: 
            self.situacao.append("Stable")
            self.gravidade.append(4)
        elif gravity_status <=42.5: 
            self.situacao.append("Potentially Stable")
            self.gravidade.append(3)
        elif gravity_status <= 84: 
            self.situacao.append("Unstable")
            self.gravidade.append(2)
        else:
            self.situacao.append("Critical")
            self.gravidade.append(1)

        
        # You can print or return the status if needed
        print(f"ID: {id}, Pressure: {pressure}, Pulse: {pulse}, Respiration: {respiration}, Status: {self.situacao[-1]}, Gravidade: {self.gravidade[-1]}")

        with open(self.new_gravity, "a", newline="") as arquivo_csv:
            # Cria um objeto escritor CSV
            escritor_csv = csv.writer(arquivo_csv)
            new_linha = []
            new_linha.append(id)
            new_linha.append(self.gravidade[-1])
            escritor_csv.writerow(new_linha)     

    def fuzzy(self, vs_file):
        with open(self.new_gravity, "w", newline="") as arquivo_csv:
            pass  # Empty block just to create an empty file
        # Step 1: Read Data
        # vs_file = os.path.join("datasets", "data_20x20_42vic", "sinais_vitais.txt" )
        number_of_victims = 0
        with open(vs_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                number_of_victims += 1
                self.id_input.append(float(row[2]))
                self.qpa_input.append(float(row[5]))
                self.pulso_input.append(float(row[6]))
                self.respiracao_input.append(float(row[7]))
                self.classe_gravidade.append(float(row[9]))
                

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
        self.simulator = ctrl.ControlSystemSimulation(system)

        for i in range(number_of_victims):
            #print(qpa_input[i], pulso_input[i], respiracao_input[i])
            pressure_value = self.qpa_input[i]
            pulse_value = self.pulso_input[i]
            respiration_value = self.respiracao_input[i]
            id_value = self.id_input[i]
            self.determine_status(pressure_value, pulse_value, respiration_value, id_value)

        statustxt = "status.csv"
        with open(statustxt, "w", newline="") as arquivo_csv:
            # Cria um objeto escritor CSV
            escritor_csv = csv.writer(arquivo_csv)
            
            i = 0
            for status in self.situacao:
                new_linha = []
                new_linha.append(int(self.id_input[i]))
                if status == "Critical":
                    new_linha.append("Critical")
                if status == "Unstable":
                    new_linha.append("Unstable")
                if status == "Potentially Stable":
                    new_linha.append("Potentially Stable")
                if status == "Stable":
                    new_linha.append("Stable")
                escritor_csv.writerow(new_linha)        
                i += 1 


            
            
        precision = precision_score(self.classe_gravidade, self.gravidade, average='weighted')
        recall = recall_score(self.classe_gravidade, self.gravidade, average='weighted')
        f1 = f1_score(self.classe_gravidade, self.gravidade, average='weighted')
        accuracy = accuracy_score(self.classe_gravidade, self.gravidade)

        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F-measure: {f1}")
        print(f"Acurácia: {accuracy}")

        num_folds = 5
        kf = KFold(n_splits=num_folds)

        # Dados de entrada
        X = np.column_stack((self.qpa_input, self.pulso_input, self.respiracao_input))

        # Classes reais
        y = self.classe_gravidade