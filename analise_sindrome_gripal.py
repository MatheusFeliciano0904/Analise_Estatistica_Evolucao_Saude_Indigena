import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, norm
import math
import os

def run_analysis():
    """
    Executa a análise completa dos dados de Síndrome Gripal em SP (2023 e 2024),
    de forma semelhante ao script de COVID, mas adaptado para:
      - Idade
      - Sintomas
      - Evolução do caso
    """
    # --- CONFIGURAÇÕES INICIAIS ---
    DATA_DIR = "dados"          # ajuste se os arquivos estiverem em outro caminho
    OUTPUT_DIR = "resultados"  # pasta onde os gráficos serão salvos
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Carregando os dados de Síndrome Gripal...")

    # Nomes dos arquivos (ajuste se forem diferentes)
    path_2023 = os.path.join(DATA_DIR, "Notificações de Síndrome Gripal_SP_2023.csv")
    path_2024 = os.path.join(DATA_DIR, "Notificações de Síndrome Gripal_SP_2024.csv")

    try:
        df_2023 = pd.read_csv(path_2023, sep=";", encoding="latin1", engine="python", on_bad_lines="skip")
        df_2024 = pd.read_csv(path_2024, sep=";", encoding="latin1", engine="python", on_bad_lines="skip")
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado. Verifique o caminho: {e.filename}")
        return

    print("Arquivos carregados com sucesso.")
    print(f"2023: {df_2023.shape[0]} linhas, {df_2023.shape[1]} colunas")
    print(f"2024: {df_2024.shape[0]} linhas, {df_2024.shape[1]} colunas\n")

    # --- ETAPA 1: Padronizar nomes de colunas e adicionar o ano ---
    def preparar_ano(df, ano):
        # normaliza nomes de colunas: minúsculo, sem acento, sem espaços
        df = df.copy()
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.normalize("NFKD")
            .str.encode("ascii", "ignore")
            .str.decode("ascii")
        )
        df["ano"] = ano
        return df

    df_2023 = preparar_ano(df_2023, 2023)
    df_2024 = preparar_ano(df_2024, 2024)

    # --- ETAPA 2: Unificar as bases ---
    df_final = pd.concat([df_2023, df_2024], ignore_index=True)

    # conferindo colunas importantes
    print("Colunas disponíveis após padronização:")
    print(df_final.columns.tolist(), "\n")

    # garantir que algumas colunas-chave existam
    col_idade = "idade"
    col_sexo = "sexo"
    col_evolucao = "evolucaocaso"        # virou tudo minúsculo e sem acento
    col_sintomas = "sintomas"
    col_data_notif = "datanotificacao"   # se existir

    # converter idade para numérico
    if col_idade in df_final.columns:
        df_final[col_idade] = pd.to_numeric(df_final[col_idade], errors="coerce")

    # converter data de notificação para datetime (se existir)
    if col_data_notif in df_final.columns:
        df_final[col_data_notif] = pd.to_datetime(df_final[col_data_notif], errors="coerce")

    # criar coluna de número de sintomas (contando itens separados por vírgula)
    if col_sintomas in df_final.columns:
        df_final["num_sintomas"] = (
            df_final[col_sintomas]
            .fillna("")
            .apply(lambda x: len([s for s in str(x).split(",") if s.strip() != ""]))
        )
    else:
        df_final["num_sintomas"] = pd.NA

    print("Dados preparados com sucesso.\n")

    # -----------------------------------------------------------------
    # --- ETAPA 3: Estatística Descritiva (semelhante ao bloco icu) ---
    # -----------------------------------------------------------------
    print("--- Análise de Estatística Descritiva (todos os anos) ---")

    desc_geral = df_final[[col_idade, "num_sintomas"]].describe()
    print(desc_geral, "\n")

    print("--- Estatística Descritiva por ano (idade) ---")
    desc_por_ano = df_final.groupby("ano")[col_idade].describe()
    print(desc_por_ano, "\n")

    # -----------------------------------------------------------------
    # --- ETAPA 4: Gráficos ---
    # -----------------------------------------------------------------

    # 4.1 Histograma de idade
    plt.figure(figsize=(10, 6))
    sns.histplot(df_final[col_idade].dropna(), bins=30, kde=True)
    plt.title("Distribuição da Idade dos Casos de Síndrome Gripal (SP, 2023-2024)")
    plt.xlabel("Idade (anos)")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "histograma_idade.png"))
    plt.close()
    print("Gráfico 'histograma_idade.png' salvo.\n")

    # 4.2 Boxplot de idade por ano
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_final, x="ano", y=col_idade)
    plt.title("Distribuição da Idade por Ano (Síndrome Gripal - SP)")
    plt.xlabel("Ano")
    plt.ylabel("Idade (anos)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "boxplot_idade_ano.png"))
    plt.close()
    print("Gráfico 'boxplot_idade_ano.png' salvo.\n")
    
    # ---------------------------------------------------
# 4.3 Gráfico de distribuição de sexo por ano
# ---------------------------------------------------

    if col_sexo in df_final.columns:
        plt.figure(figsize=(10, 6))
        sexo_ano = df_final.groupby(["ano", col_sexo]).size().reset_index(name="contagem")

        sns.barplot(
            data=sexo_ano,
            x="ano",
            y="contagem",
            hue=col_sexo,
            palette="Set2"
        )

        plt.title("Distribuição de Casos por Sexo (2023 x 2024)")
        plt.xlabel("Ano")
        plt.ylabel("Número de Notificações")
        plt.legend(title="Sexo")
        plt.tight_layout()

        plt.savefig(os.path.join(OUTPUT_DIR, "grafico_sexo_por_ano.png"))
        plt.close()
        print("Gráfico 'grafico_sexo_por_ano.png' salvo.\n")

    # -----------------------------------------------------------------
    # --- ETAPA 5: Testes de Hipótese (2023 vs 2024) ---
    # -----------------------------------------------------------------
    print("--- Testes de Hipótese: 2023 vs 2024 ---")

    df_23 = df_final[df_final["ano"] == 2023].copy()
    df_24 = df_final[df_final["ano"] == 2024].copy()

    # 5.1 Teste t para idade média
    idade_23 = df_23[col_idade].dropna()
    idade_24 = df_24[col_idade].dropna()

    print("\n>>> Teste t para idade média (2023 vs 2024)")
    print(f"Média idade 2023: {idade_23.mean():.2f}")
    print(f"Média idade 2024: {idade_24.mean():.2f}")

    t_stat, p_val = ttest_ind(idade_23, idade_24, equal_var=False)
    print(f"Estatística t: {t_stat:.4f}, P-valor: {p_val:.4f}")

    if p_val < 0.05:
        print("Conclusão: Rejeitamos H0 → diferença significativa na idade média entre 2023 e 2024.")
    else:
        print("Conclusão: Não rejeitamos H0 → não há evidência de diferença na idade média.\n")

    # 5.2 Teste de proporção para um sintoma específico
    if col_sintomas in df_final.columns:
        SINTOMA_ALVO = "Tosse"   # você pode trocar para "Coriza", "Febre", etc.
        print(f"\n>>> Teste de proporção para o sintoma: {SINTOMA_ALVO}")

        has_symp_23 = df_23[col_sintomas].str.contains(SINTOMA_ALVO, case=False, na=False)
        has_symp_24 = df_24[col_sintomas].str.contains(SINTOMA_ALVO, case=False, na=False)

        x1 = has_symp_23.sum()
        n1 = has_symp_23.shape[0]
        x2 = has_symp_24.sum()
        n2 = has_symp_24.shape[0]

        p1 = x1 / n1
        p2 = x2 / n2

        print(f"Proporção de casos com {SINTOMA_ALVO} em 2023: {p1:.4f}")
        print(f"Proporção de casos com {SINTOMA_ALVO} em 2024: {p2:.4f}")

        p_pool = (x1 + x2) / (n1 + n2)
        se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        z = (p1 - p2) / se
        p_z = 2 * (1 - norm.cdf(abs(z)))

        print(f"Estatística z: {z:.4f}, P-valor: {p_z:.4f}")

        if p_z < 0.05:
            print("Conclusão: Rejeitamos H0 → a proporção desse sintoma difere entre 2023 e 2024.")
        else:
            print("Conclusão: Não rejeitamos H0 → proporções semelhantes.\n")

    # -----------------------------------------------------------------
    # --- ETAPA 6: Análise de Probabilidade ---
    # -----------------------------------------------------------------
    print("\n--- Análise de Probabilidade (todas as notificações) ---")

    # Exemplo: probabilidade de idade ≥ 35
    idosos = df_final[df_final[col_idade] >= 35]
    total_validos = df_final[col_idade].notna().sum()
    prob_idoso = len(idosos) / total_validos if total_validos > 0 else 0

    print(f"Probabilidade de um caso ter idade ≥ 35 anos: {prob_idoso:.2%}")

    # Exemplo: probabilidade de ter "muitos sintomas" (acima do percentil 75 de num_sintomas)
    if df_final["num_sintomas"].notna().any():
        p75 = df_final["num_sintomas"].quantile(0.75)
        acima_p75 = df_final[df_final["num_sintomas"] > p75].shape[0]
        total_ns = df_final["num_sintomas"].notna().sum()
        prob_muitos_sintomas = acima_p75 / total_ns if total_ns > 0 else 0

        print(f"Percentil 75 do número de sintomas: {p75:.2f}")
        print(f"Probabilidade de um caso ter mais sintomas que esse limiar: {prob_muitos_sintomas:.2%}")

    print("\nAnálise concluída. Verifique os arquivos de saída na pasta 'resultados'.")


if __name__ == "__main__":
    run_analysis()
