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
    # Perguntas a serem respondidas:
    # 1. Qual a distribuição de idade dos casos notificados em 2022 e 2024?
    # 2. Quais os sintomas mais comuns em cada ano?
    # 3. Qual a taxa de evolução (cura, cancelado(óbito), ignorado) em cada ano?
    # 4. Há diferenças estatísticas significativas entre os anos para idade, sintomas e evolução?
    # 5. Visualizações gráficas para comparar os anos.
    # --- FIM DAS PERGUNTAS ---
    
    #  1 ** --- CONFIGURAÇÕES INICIAIS ---
    dados = "dados"          # pasta onde estão os CSV
    output = "output"   # pasta onde os gráficos serão salvos
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
    
    # Exibir linhas para conferência
    print("Amostra dos dados 2022:")
    print(df_amostra_2022, "\n")
    print("Amostra dos dados 2024:")
    print(df_amostra_2024, "\n")

    
    # Exibir informações básicas dos DataFrames
    print("Informações do DataFrame 2022:")
    print(df_amostra_2022.info(), "\n")
    print("Informações do DataFrame 2024:")
    print(df_amostra_2024.info(), "\n")
    
    
    
    
    # 2 ** --- TRATAMENTO DE BASES ---
    def preparar_ano(df, ano):
        
        df = df.copy() # cria uma cópia para evitar modificar o original
        df.columns = (
            df.columns # normaliza nomes de colunas para cada DataFrame: minúsculo, sem acento, sem espaços
            .str.strip() # remove espaços em branco nas extremidades
            .str.lower() # converte para minúsculo
            .str.normalize("NFKD") # normaliza para decompor caracteres acentuados
            .str.encode("ascii", "ignore") # remove caracteres não ASCII, por exemplo acentos serão ignorados
            .str.decode("ascii") # decodifica de volta para string, por exemplo 'ã' vira 'a'
        )
        
        df["ano"] = ano # adiciona coluna com o ano correspondente, logo, 2022 ou 2024
        return df # retorna o DataFrame modificado

    #converter idade para int64
    df_amostra_2022['idade'] = pd.to_numeric(df_amostra_2022['idade'], errors='coerce').astype('Int64')
    df_amostra_2024['idade'] = pd.to_numeric(df_amostra_2024['idade'], errors='coerce').astype('Int64')
    
    # ver como ficaram as coluna 
    print("Colunas do DataFrame 2022 após normalização:")
    print(df_amostra_2022.info(), "\n")
    print("Colunas do DataFrame 2024 após normalização:")
    print(df_amostra_2024.info(), "\n")
    
    # aqui corrigimos: 2022 e 2024
    df_amostra_2022 = preparar_ano(df_amostra_2022, 2022)
    df_amostra_2024 = preparar_ano(df_amostra_2024, 2024)
    
    #Ver como ficaram as colunas
    
    print("Colunas do DataFrame 2022 após normalização:")
    print(df_amostra_2022.info(), "\n")
    print("Colunas do DataFrame 2024 após normalização:")
    print(df_amostra_2024.info(), "\n")
        
    # Unificar as bases
    df_unificado = pd.concat([df_amostra_2022, df_amostra_2024], ignore_index=True) 
    
    # Exibir as primeiras linhas do DataFrame unificado para conferência
    print("Amostra dos dados unificados (2022 e 2024):")
    display(df_unificado, "\n")


    # Agora analisar variáveis de interesse: idade, sintomas, evolução do caso
    # vamos criar UM DATAFRAME só com as colunas de interesse para facilitar a análise
    colunas_interesse = [
        "idade",  # idade do paciente
        "sintomas",  # sintomas reportados (ex: febre, tosse, dor de garganta)
        "evolucaocaso",  # evolução do caso (ex: cura, internação, óbito)
        "ano"
    ]
    
    # Criar DataFrame só com colunas de interesse
    df_analise = df_unificado[colunas_interesse].copy()
    
    print("Amostra dos dados para análise (Com as colunas de interresse):")
    display(df_analise, "\n")
        
    # 3 ** -- Estatística descritiva --
    
    # Estatísticas descritivas para idade por ano
    estatisticas_idade = df_analise.groupby("ano")["idade"].describe()
    print("Estatísticas descritivas para idade por ano:")
    display(estatisticas_idade, "\n")
    
    # Estatísticas descritivas para sintomas por ano, mostrando separado para cada ano ( 4 mais comuns).
    sintomas_mais_comuns = (
        df_analise
        .groupby(["ano", "sintomas"])
        .size()
        .reset_index(name="contagem")
        .sort_values(by=["ano", "contagem"], ascending=[True, False])
        .groupby("ano")
        .head(4)
    )
    print("Sintomas mais comuns por ano:")
    display(sintomas_mais_comuns, "\n")
    
    # Estatísticas descritivas para evolução do caso por ano
    evolucao_caso = (
        df_analise
        .groupby(["ano", "evolucaocaso"])
        .size()
        .reset_index(name="contagem")
        .sort_values(by=["ano", "contagem"], ascending=[True, False])
    )
    print("Evolução do caso por ano:")
    display(evolucao_caso, "\n")

    
    # # 4 ** -- Probabilidade --
    
    # PROBABILIDADE: Qual a probabilidade de um caso evoluir para óbito em 2022 vs 2024?
    def calcular_probabilidade_obito(df, ano):
        df_ano = df[df["ano"] == ano]
        total_casos = len(df_ano)
        casos_obito = len(df_ano[df_ano["evolucaocaso"] == "Cancelado"])
        probabilidade = casos_obito / total_casos if total_casos > 0 else 0
        return probabilidade
    prob_obito_2022 = calcular_probabilidade_obito(df_analise, 2022)
    prob_obito_2024 = calcular_probabilidade_obito(df_analise, 2024)
    
    print(f"Probabilidade de óbito em 2022: {prob_obito_2022:.4f}")
    print(f"Probabilidade de óbito em 2024: {prob_obito_2024:.4f}\n")
    
    #PROBAILIDADE: Qual a probabilidade de um caso curar em 2022 vs 2024?
    def calcular_probabilidade_cura(df, ano):
        df_ano = df[df["ano"] == ano]
        total_casos = len(df_ano)
        casos_cura = len(df_ano[df_ano["evolucaocaso"] == "Cura"])
        probabilidade = casos_cura / total_casos if total_casos > 0 else 0
        return probabilidade
    prob_cura_2022 = calcular_probabilidade_cura(df_analise, 2022)
    prob_cura_2024 = calcular_probabilidade_cura(df_analise, 2024)
    print(f"Probabilidade de cura em 2022: {prob_cura_2022:.4f}")
    print(f"Probabilidade de cura em 2024: {prob_cura_2024:.4f}\n")
    
    #PROBAILIDADE: Qual a probabilidade de um caso ser ignorado em 2022 vs 2024?
    def calcular_probabilidade_ignorado(df, ano):
        df_ano = df[df["ano"] == ano]
        total_casos = len(df_ano)
        casos_ignorado = len(df_ano[df_ano["evolucaocaso"] == "Ignorado"])
        probabilidade = casos_ignorado / total_casos if total_casos > 0 else 0
        return probabilidade
    prob_ignorado_2022 = calcular_probabilidade_ignorado(df_analise, 2022)
    prob_ignorado_2024 = calcular_probabilidade_ignorado(df_analise, 2024)
    print(f"Probabilidade de ignorado em 2022: {prob_ignorado_2022:.4f}")
    print(f"Probabilidade de ignorado em 2024: {prob_ignorado_2024:.4f}\n")
    
    
    
    # # 5 ** -- Inferencia estatística --
    
    # inferência estatística para idade entre 2022 e 2024
    
    # amostras de idade por ano
    idades_2022 = df_analise.loc[df_analise["ano"] == 2022, "idade"].dropna()
    idades_2024 = df_analise.loc[df_analise["ano"] == 2024, "idade"].dropna()
    
        # 1. Teste Médias Idade
        
    idade_media_2022 = df_analise[df_analise["ano"] == 2022]["idade"].mean()
    idade_media_2024 = df_analise[df_analise["ano"] == 2024]["idade"].mean()
    print(f"Média de idade em 2022: {idade_media_2022:.2f}")
    print(f"Média de idade em 2024: {idade_media_2024:.2f}\n")
    
    
    # definir hipótese nula e alternativa
    # H0: A média de idade em 2022 é igual à média de idade em 2024.
    # H1: A média de idade em 2022 é diferente da média de idade em 2024.
    valor_base = 0.05  # nível de significância
    
    #boxplot para idade (exportar gráfico para pasta resultados)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="ano", y="idade", data=df_analise)
    plt.title("Boxplot da Idade por Ano")
    plt.savefig(os.path.join(output, "boxplot_idade_por_ano.png"))
    plt.close()
    
    # Histograma para idade tanto 2022 quanto 2024 (exportar gráfico para pasta resultados)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(idades_2022, bins=30, kde=True, color='blue')
    plt.title("Histograma da Idade em 2022")
    plt.savefig(os.path.join(output, "histograma_idade_2022.png"))
    plt.close()
    plt.figure(figsize=(10, 6))
    sns.histplot(idades_2024, bins=30, kde=True, color='orange')
    plt.title("Histograma da Idade em 2024")
    plt.savefig(os.path.join(output, "histograma_idade_2024.png"))
    plt.close()

    
    # qqplot para idade (exportar gráfico para pasta resultados)
    import statsmodels.api as sm
    plt.figure(figsize=(10, 6))
    sm.qqplot(idades_2022, line ='s')
    plt.title("QQ Plot da Idade em 2022")
    plt.savefig(os.path.join(output, "qqplot_idade_2022.png"))
    plt.close()
    plt.figure(figsize=(10, 6))
    sm.qqplot(idades_2024, line ='s')
    plt.title("QQ Plot da Idade em 2024")
    plt.savefig(os.path.join(output, "qqplot_idade_2024.png"))
    plt.close()
    
    
    # teste shapiro-wilk para normalidade
    from scipy.stats import shapiro
    stat_2022, p_2022 = shapiro(idades_2022)
    stat_2024, p_2024 = shapiro(idades_2024)
    print(f"Teste Shapiro-Wilk para 2022: estatística={stat_2022:.4f}, p-valor={p_2022:.4f}")
    print(f"Teste Shapiro-Wilk para 2024: estatística={stat_2024:.4f}, p-valor={p_2024:.4f}\n")
    # interpretar resultados
    alpha = 0.05
    if p_2022 > alpha:
        print("2022: A amostra parece vir de uma distribuição normal (falha em rejeitar H0)\n")
    else:
        print("2022: A amostra não parece vir de uma distribuição normal (rejeita H0)\n")
        
    if p_2024 > alpha:
        print("2024: A amostra parece vir de uma distribuição normal (falha em rejeitar H0)\n")
    else:
        print("2024: A amostra não parece vir de uma distribuição normal (rejeita H0)\n")
    
    # realizar o teste t
    stat, p_value = ttest_ind(idades_2022, idades_2024, equal_var=False)
    print(f"Teste t para idade entre 2022 e 2024: estatística={stat:.4f}, p-valor={p_value:.4f}\n")
    if p_value < valor_base:
        print("Rejeita H0: Há diferença significativa na idade média entre 2022 e 2024.\n")
    else:
        print("Falha em rejeitar H0: Não há diferença significativa na idade média entre 2022 e 2024.\n")  
    
    
        
if __name__ == "__main__":
    run_analysis()       

