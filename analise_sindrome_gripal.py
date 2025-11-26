from IPython.display import display # biblioteca para exibir DataFrames 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, norm
import math # biblioteca para funções matemáticas
import os # biblioteca para manipulação de caminhos e pastas

def run_analysis():
    """
    Executa a análise completa dos dados de Síndrome Gripal em SP (2022 e 2024),
    de forma semelhante ao script de COVID, mas adaptado para:
      - Idade
      - Sintomas
      - Evolução do caso
    """
    
    # --- CONFIGURAÇÕES INICIAIS ---
    dados = "dados"          # pasta onde estão os CSV
    output = "resultados"   # pasta onde os gráficos serão salvos
    os.makedirs(output, exist_ok=True) # cria a pasta resultados, se não existir ainda.

    # Nomes dos arquivos (dentro da pasta 'dados')
    
    path_2022 = os.path.join(dados, "Notificações de Síndrome Gripal - 2022_MAIOR.csv")
    path_2024 = os.path.join(dados, "Notificações de Síndrome Gripal - 2024_MAIOR.csv")

    try: #  ler os aruqivos CSV e caso de erro, gera uma mensagem erro. Vou ler só 5000 linhas para teste(nrows).
        df_amostra_2022 = pd.read_csv(path_2022, sep=";", encoding="latin1",engine="python", on_bad_lines="skip", nrows=5000)
        df_amostra_2024 = pd.read_csv(path_2024, sep=";", encoding="latin1",engine="python", on_bad_lines="skip", nrows=5000)
    except FileNotFoundError as e:
        print(f"Erro ao carregar os arquivos: {e}")
        return
    
    print("Carregamento dos dados concluído.")
    
    # Exibir as primeiras linhas para conferência
    print("Amostra dos dados 2022:")
    print(df_amostra_2022.head(), "\n")
    print("Amostra dos dados 2024:")
    print(df_amostra_2024.head(), "\n")


    # Mostrar as colunas em listas disponíveis em cada DataFrame
    
    print("Colunas disponíveis em 2022:")
    print(df_amostra_2022.columns.tolist(), "\n")
    print("Colunas disponíveis em 2024:")
    print(df_amostra_2024.columns.tolist(), "\n")
    
    # --- TRATAMENTO DE BASES ---
    def preparar_ano(df, ano):
        
        df = df.copy() # cria uma cópia para evitar modificar o original
        df.columns = (
            df.columns # normaliza nomes de colunas para cada DataFrame: minúsculo, sem acento, sem espaços
            .str.strip() # remove espaços em branco nas extremidades
            .str.lower() # converte para minúsculo
            .str.normalize("NFKD") # normaliza para decompor caracteres acentuados
            .str.encode("ascii", "ignore") # remove caracteres não ASCII, por exemplo acentos serão ignorados
            .str.decode("ascii") # decodifica de volta para string, por exemplo 'ã' vira 'a'
            .str.replace(" ", "_")  # substitui espaços por underline por exemplo 'Data Notificação' vira 'data_notificacao'
        )
        
        df["ano"] = ano # adiciona coluna com o ano correspondente, logo, 2022 ou 2024
        return df # retorna o DataFrame modificado

    # aqui corrigimos: 2022 e 2024
    df_amostra_2022 = preparar_ano(df_amostra_2022, 2022)
    df_amostra_2024 = preparar_ano(df_amostra_2024, 2024)
    
    # Unificar as bases
    df_unificado = pd.concat([df_amostra_2022, df_amostra_2024], ignore_index=True) 
    
    # Exibir as primeiras linhas do DataFrame unificado para conferência
    print("Amostra dos dados unificados (2022 e 2024):")
    display(df_unificado.head(), "\n")
    
if __name__ == "__main__":
    run_analysis()       

