"""
Etapa 1 - Gera o CSV para treinar a arvore de decisao.

Pergunta: "A escola tem infraestrutura digital basica?"
Alvo (infra_digital_basica) = 1 se IN_INTERNET_ALUNOS == 1
                              E a escola tem desktop, portatil OU tablet para os alunos
                              (IN_DESKTOP_ALUNO, IN_COMP_PORTATIL_ALUNO, IN_TABLET_ALUNO).
                              Senao, 0.
Obs.: usamos as colunas IN_ (0/1) em vez das QT_ porque as QT_ podem
conter o codigo 88888 (valor extremo), segundo o dicionario do INEP.

Como usar:
1. Coloque o caminho do arquivo Tabela_Escola_2025_V2.csv em CAMINHO_ENTRADA.
2. Rode:  python etapa1_gerar_csv.py
3. Copie a saida que aparecer no terminal e mande para a Claude.
"""

import pandas as pd

# ---------------------------------------------------------------
# 1) Configuracao (mude so o CAMINHO_ENTRADA)
# ---------------------------------------------------------------
CAMINHO_ENTRADA = "Tabela_Escola_2025_V2.csv"
CAMINHO_SAIDA = "dados_etapa1.csv"

COL_ID = "CO_ENTIDADE"
COLS_FILTRO = ["SG_UF", "TP_SITUACAO_FUNCIONAMENTO"]
COLS_PISTAS = [
    "IN_LABORATORIO_INFORMATICA",
    "IN_BIBLIOTECA",
    "TP_LOCALIZACAO",
    "TP_DEPENDENCIA",
    "IN_ENERGIA_GERADOR_FOSSIL",
]
COLS_ALVO = [
    "IN_INTERNET_ALUNOS",
    "IN_DESKTOP_ALUNO",
    "IN_COMP_PORTATIL_ALUNO",
    "IN_TABLET_ALUNO",
]

# ---------------------------------------------------------------
# 2) Leitura (so as colunas que interessam)
# Os arquivos do INEP usam ';' como separador e codificacao latin-1.
# ---------------------------------------------------------------
colunas = [COL_ID] + COLS_FILTRO + COLS_PISTAS + COLS_ALVO
df = pd.read_csv(
    CAMINHO_ENTRADA,
    sep=";",
    encoding="latin-1",
    usecols=colunas,
    low_memory=False,
)
print(f"Linhas lidas (Brasil inteiro): {len(df)}")

# ---------------------------------------------------------------
# 3) Filtro das linhas: Amazonas + escolas em atividade
# (confirme no dicionario do INEP que 1 = "Em atividade")
# ---------------------------------------------------------------
df = df[(df["SG_UF"] == "AM") & (df["TP_SITUACAO_FUNCIONAMENTO"] == 1)].copy()
print(f"Linhas apos o filtro (AM, em atividade): {len(df)}")

# ---------------------------------------------------------------
# 4) Valores vazios: so reportamos, sem esconder
# ---------------------------------------------------------------
print("\nValores vazios por coluna (apos o filtro):")
print(df[COLS_PISTAS + COLS_ALVO].isna().sum())

# Nas colunas que definem o alvo, vazio = escola nao informou = tratamos como 0
df[COLS_ALVO] = df[COLS_ALVO].fillna(0)

# Nas pistas, removemos linhas com vazio (e dizemos quantas sairam)
antes = len(df)
df = df.dropna(subset=COLS_PISTAS)
print(f"\nLinhas removidas por vazio nas pistas: {antes - len(df)}")

# ---------------------------------------------------------------
# 5) Criar o alvo 0/1
# ---------------------------------------------------------------
tem_aparelho = (
    (df["IN_DESKTOP_ALUNO"] == 1)
    | (df["IN_COMP_PORTATIL_ALUNO"] == 1)
    | (df["IN_TABLET_ALUNO"] == 1)
)
tem_internet = df["IN_INTERNET_ALUNOS"] == 1
df["infra_digital_basica"] = (tem_internet & tem_aparelho).astype(int)

# ---------------------------------------------------------------
# 6) Salvar: id + pistas + alvo (as colunas do alvo NAO vao como pistas)
# ---------------------------------------------------------------
saida = df[[COL_ID] + COLS_PISTAS + ["infra_digital_basica"]]
saida.to_csv(CAMINHO_SAIDA, index=False, encoding="utf-8")

print(f"\nArquivo salvo: {CAMINHO_SAIDA}  (formato: {saida.shape[0]} linhas x {saida.shape[1]} colunas)")
print("\nDistribuicao do alvo (0 = nao, 1 = sim):")
print(saida["infra_digital_basica"].value_counts())
print(saida["infra_digital_basica"].value_counts(normalize=True).round(3))
print("\nPrimeiras linhas:")
print(saida.head())
