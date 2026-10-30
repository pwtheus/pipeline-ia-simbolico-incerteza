"""
Tabelas cruzadas das 5 pistas contra o alvo (infra_digital_basica).

Use para fazer o calculo de entropia e ganho A MAO.
So depois de calcular, mude MOSTRAR_GANHO para True e confira.

Para olhar os niveis seguintes da arvore, preencha FILTRO.
Exemplo: FILTRO = {"IN_LABORATORIO_INFORMATICA": 1}
mostra as tabelas so das escolas COM laboratorio (2o nivel).
Para voltar a raiz, use FILTRO = {}.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------
MOSTRAR_GANHO = False   # mude para True so depois de calcular a mao
FILTRO = {}             # {} = raiz da arvore (todas as escolas)

ALVO = "infra_digital_basica"
PISTAS = [
    "IN_LABORATORIO_INFORMATICA",
    "IN_BIBLIOTECA",
    "TP_LOCALIZACAO",
    "TP_DEPENDENCIA",
    "IN_ENERGIA_GERADOR_FOSSIL",
]


def entropia(contagens):
    """Entropia (em bits) de um grupo, dado o numero de escolas por classe."""
    total = sum(contagens)
    if total == 0:
        return 0.0
    p = np.array(contagens) / total
    p = p[p > 0]  # evita log2(0)
    return float(-(p * np.log2(p)).sum())


# ---------------------------------------------------------------
# Leitura e filtro
# ---------------------------------------------------------------
df = pd.read_csv("dados_etapa1.csv")

for coluna, valor in FILTRO.items():
    df = df[df[coluna] == valor]

print(f"Escolas neste no da arvore: {len(df)}")
contagem_alvo = df[ALVO].value_counts().reindex([0, 1], fill_value=0)
print(f"Classe 0: {contagem_alvo[0]}  |  Classe 1: {contagem_alvo[1]}")

h_pai = entropia(contagem_alvo.tolist())
if MOSTRAR_GANHO:
    print(f"Entropia deste no: {h_pai:.4f}")

# ---------------------------------------------------------------
# Tabela cruzada de cada pista (as pistas ja usadas no FILTRO saem da lista)
# ---------------------------------------------------------------
for pista in PISTAS:
    if pista in FILTRO:
        continue

    tabela = pd.crosstab(df[pista], df[ALVO]).reindex(columns=[0, 1], fill_value=0)
    print("\n" + "=" * 50)
    print(f"Pista: {pista}")
    print(tabela)

    if MOSTRAR_GANHO:
        total = tabela.values.sum()
        media_filhos = 0.0
        for valor, linha in tabela.iterrows():
            h_filho = entropia(linha.tolist())
            peso = linha.sum() / total
            media_filhos += peso * h_filho
            print(f"  valor {valor}: entropia = {h_filho:.4f}  (peso = {peso:.4f})")
        print(f"  Ganho de informacao = {h_pai - media_filhos:.4f}")
