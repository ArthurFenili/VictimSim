import pandas as pd

def remove_colunas_por_numero(arquivo_entrada, arquivo_saida, numeros_colunas_para_remover):
    # Carrega o arquivo CSV para um DataFrame sem header
    df = pd.read_csv(arquivo_entrada, header=None)

    # Remove as colunas especificadas pelos números
    df = df.drop(columns=numeros_colunas_para_remover, errors='ignore')

    # Salva o DataFrame modificado em um novo arquivo CSV
    df.to_csv(arquivo_saida, index=False, header=False)

if __name__ == "__main__":
    # Substitua 'input.csv', 'output.csv' e [1, 3] pelos seus valores
    arquivo_entrada = 'sinais_vitais_teste.txt'
    arquivo_saida = 'sinais_vitais_teste1.txt'
    numeros_colunas_para_remover = [0]

    

    # Chama a função para remover as colunas
    remove_colunas_por_numero(arquivo_entrada, arquivo_saida, numeros_colunas_para_remover)

    print(f'Colunas removidas com sucesso. Novo arquivo: {arquivo_saida}')
