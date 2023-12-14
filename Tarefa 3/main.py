import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from joblib import dump, load


# Define the file path
file_path = 'sinais800_normalizados.txt'
file_model1 = "model1.pickle" #11.67
file_model2 = "model2.joblib" #6
file_model3 = "model3.pickle" #7.94
test_data_file = 'sinais_teste_normalizados.txt'
with open(test_data_file, 'r') as file:
    lines = file.readlines()

#ignore the first line
lines = lines[1:]

# Extract data from lines and create a data matrix
teste_sala = []
for line in lines:
    # Assuming data is comma-separated, modify as needed
    row = [float(value) for value in line.strip().split(sep=',')]
    teste_sala.append(row)
# Convert the list of lists into a NumPy array
teste_sala = np.array(teste_sala)
print(teste_sala)

try:
    # Read the TXT file and create a list of lines
    with open(file_path, 'r') as file:
        lines = file.readlines()

    #ignore the first line
    lines = lines[1:]

    # Extract data from lines and create a data matrix
    data_matrix = []
    for line in lines:
        # Assuming data is comma-separated, modify as needed
        row = [float(value) for value in line.strip().split(sep=',')]
        data_matrix.append(row)

    # Convert the list of lists into a NumPy array
    data_matrix = np.array(data_matrix)

    # Print the data matrix
    print("Data Matrix:")
    print(data_matrix)

except FileNotFoundError:
    print(f"The file '{file_path}' does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")


# Assuming the last column is the target variable
features = data_matrix[:, :-1]
target = data_matrix[:, -1]

#create a simple tkinter ui to select which model to run
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# Create the root window
root = tk.Tk()
root.title("Model Selection")
root.geometry("300x200")

# Create a label and a combobox
ttk.Label(root, text="Select a model:").pack(pady=10)
model = tk.StringVar()
model_combobox = ttk.Combobox(root, width=27, textvariable=model)
model_combobox['values'] = ("Model 1", "Model 2", "Model 3")
model_combobox.pack()

# Create a button to run the selected model
def run_model():
    if model.get() == "Model 1":
        run_model1()
    elif model.get() == "Model 2":
        run_model2()
    elif model.get() == "Model 3":
        run_model3()
    else:
        messagebox.showerror("Model Selection", "Please select a model")

ttk.Button(root, text="Run Model", command=run_model).pack(pady=50)


def run_model1():
    # MODELO 1 
    # 500 EPOCAS E BATCH SIZE 16 
    # 30% PARA VALIDAÇÃO E 70% PARA TREINO
    # OPTIMIZER ADAM
    # LOSS FUNCTION MEAN SQUARED ERROR
    # 3 HIDDEN LAYERS COM 64, 32 E 16 NEURONIOS
    # ACTIVATION FUNCTION RELU, SIGMOID E RELU

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    #Pelos testes, as funcoes de ativacao que mais se aproximaram do valor real foram a relu e a linear
    #2 relu, 4 linear, test_size =0.1, todos 64 neuronios e 500 epocas, RMSE = 43
    #4 relu, 2 linear, test_size =0.1, todos 64 neuronios e 500 epocas, RMSE = 14
    #3 relu, 3 linear, test_size =0.1, todos 64 neuronios e 500 epocas, RMSE = 19
    #5 relu, 1 linear, test_size =0.1, todos 64 neuronios e 500 epocas, RMSE = 28
    #6 relu, 1 com 64 neuronios e 5 com 32 neuronios e 500 epocas, test_size =0.2, RMSE = 30
    #6 relu, 1 com 64 neuronios e 5 com 32 neuronios e 750 epocas, test_size =0.1, RMSE = 15

    model1 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(64, activation='linear'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(16, activation='linear'),
        tf.keras.layers.Dense(1)  # Output layer 
    ])

    model1.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model1.fit(X_train_scaled, y_train, epochs=500, batch_size=32, validation_data=(X_test_scaled, y_test))

    # Evaluate the model on the test set
    mse = model1.evaluate(X_test_scaled, y_test)
    print(f"Mean Squared Error on Test Set for MODEL 1: {mse}")

    # Make predictions
    predictions = model1.predict(X_test_scaled)

    real_value = []
    predicted_value = []

    print("MODEL 1 Predictions:")
    for i in range(len(y_test)):
        predicted_value.append(predictions[i][0])
        real_value.append(y_test[i])
        print(f"Predicted value: {predictions[i][0]:.2f} | Actual value: {y_test[i]}")

    # save model
    dump(model1, file_model1)


def run_model2():
    # MODELO 2 
    # 500 EPOCAS E BATCH SIZE 16 
    # 30% PARA VALIDAÇÃO E 70% PARA TREINO
    # OPTIMIZER ADAM
    # LOSS FUNCTION MEAN SQUARED ERROR
    # 3 HIDDEN LAYERS COM 64, 32 E 16 NEURONIOS
    # ACTIVATION FUNCTION RELU, SIGMOID E RELU

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(64, activation='linear'),
        tf.keras.layers.Dense(32, activation='linear'),
        tf.keras.layers.Dense(16, activation='linear'),
        tf.keras.layers.Dense(1)  # Output layer for regression
    ])

    model2.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model2.fit(X_train_scaled, y_train, epochs=500, batch_size=16, validation_data=(X_test_scaled, y_test))

    # Evaluate the model on the test set
    mse = model2.evaluate(X_test_scaled, y_test)
    print(f"Mean Squared Error on Test Set for MODEL 2: {mse}")



    # Make predictions
    predictions = model2.predict(X_test_scaled)

    real_value = []
    predicted_value = []
    erros = []

    print("MODEL 2 Predictions:")
    for i in range(len(y_test)):
        predicted_value.append(predictions[i][0])
        real_value.append(y_test[i])
        erros.append(abs(y_test[i] - predictions[i][0]))
        print(f"Predicted value: {predictions[i][0]:.2f} | Actual value: {y_test[i]}")

    #plot error curve based on predicted and real values
    import matplotlib.pyplot as plt

    #plot error curve based on predicted and real values, where the x axis is the real value and the y axis is the error
    plt.plot(real_value, erros, 'ro')
    plt.title('Erro do valor estimado pelo modelo em relação ao valor real')
    plt.xlabel('Valor real')
    plt.ylabel('Erro')
    plt.show()

    

    teste_sala_scaled = scaler.transform(teste_sala)
    resultado_Teste_sala = model2.predict(teste_sala_scaled)
    print("Resultado Teste Cego:")
    print(resultado_Teste_sala)

    #save the result to txt
    with open('resultado.txt', 'w') as file:
        for item in resultado_Teste_sala:
            file.write("%s\n" % item)
            
    # save model
    dump(model2, file_model2)

def run_model3():
    #RMSE = 21.5
    # MODELO 3 
    # 500 EPOCAS E BATCH SIZE 16 
    # 30% PARA VALIDAÇÃO E 70% PARA TREINO
    # OPTIMIZER ADAM
    # LOSS FUNCTION MEAN SQUARED ERROR
    # 3 HIDDEN LAYERS COM 64, 32 E 16 NEURONIOS
    # ACTIVATION FUNCTION RELU, SIGMOID E RELU

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model3 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(64, activation='linear'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='linear'),
        tf.keras.layers.Dense(1)  # Output layer for regression
    ])

    model3_optimizer = Adam(learning_rate=0.001)
    model3.compile(optimizer=model3_optimizer, loss='mean_squared_error')

    # Train the model for an initial 350 epochs
    model3.fit(X_train_scaled, y_train, epochs=500, batch_size=16, validation_data=(X_test_scaled, y_test))

    # Evaluate the model on the test set
    mse = model3.evaluate(X_test_scaled, y_test)
    print(f"Mean Squared Error on Test Set for MODEL 3: {mse}")

    # Make predictions
    predictions = model3.predict(X_test_scaled)

    real_value = []
    predicted_value = []

    print("MODEL 3 Predictions:")
    for i in range(len(y_test)):
        predicted_value.append(predictions[i][0])
        real_value.append(y_test[i])
        print(f"Predicted value: {predictions[i][0]:.2f} | Actual value: {y_test[i]}")

    # save model
    dump(model3, file_model3)


# Run the mainloop
root.mainloop()