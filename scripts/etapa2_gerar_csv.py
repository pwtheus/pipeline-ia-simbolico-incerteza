#!/usr/bin/env python3
"""
Gera a base de treinamento "Densidade de Informatização" para o Estado do Amazonas
a partir do Censo Escolar 2025 (Tabela_Escola_2025_V2.csv).

Alvo:
    DENSIDADE_INFORMATIZACAO =
        (QT_DESKTOP_ALUNO + QT_COMP_PORTATIL_ALUNO + QT_TABLET_ALUNO) / QT_SALAS_UTILIZADAS

Uso:
    python scripts/etapa2_gerar_csv.py <Tabela_Escola_2025_V2.csv> [saida.csv]
"""
import sys
import numpy as np
import pandas as pd

ENTRADA = sys.argv[1] if len(sys.argv) > 1 else "Tabela_Escola_2025_V2.csv"
SAIDA = sys.argv[2] if len(sys.argv) > 2 else "dados_etapa2.csv"

UF = "AM"                 # Estado do Amazonas
MAX_SALAS = 60            # remove outliers de QT_SALAS_UTILIZADAS (máx. nacional: 945)
SENTINELA = 88888         # código "não informado" do Censo nas colunas QT_*

COLS_DISPOSITIVOS = ["QT_DESKTOP_ALUNO", "QT_COMP_PORTATIL_ALUNO", "QT_TABLET_ALUNO"]
COL_SALAS = "QT_SALAS_UTILIZADAS"
PREDITORAS = [
    "IN_INTERNET_ALUNOS",
    "IN_LABORATORIO_INFORMATICA",
    "IN_INTERNET_APRENDIZAGEM",
    "QT_EQUIP_TV",
    "QT_EQUIP_MULTIMIDIA",
    "IN_EQUIP_LOUSA_DIGITAL",
]
ALVO = "DENSIDADE_INFORMATIZACAO"

usar = ["SG_UF", "TP_SITUACAO_FUNCIONAMENTO", COL_SALAS] + COLS_DISPOSITIVOS + PREDITORAS
df = pd.read_csv(ENTRADA, sep=";", encoding="latin-1", usecols=usar, low_memory=False)
log = [("linhas na base nacional", len(df))]

# 1) Escopo: apenas o Amazonas
df = df[df["SG_UF"] == UF]
log.append((f"escolas de {UF}", len(df)))

# 2) Apenas escolas em atividade (situação 1). As demais têm infraestrutura toda vazia.
df = df[df["TP_SITUACAO_FUNCIONAMENTO"] == 1]
log.append(("em atividade", len(df)))

# 3) Código 88888 ("não informado") -> ausente, nas colunas QT_* usadas
cols_qt = [c for c in COLS_DISPOSITIVOS + [COL_SALAS] + PREDITORAS if c.startswith("QT_")]
df[cols_qt] = df[cols_qt].replace(SENTINELA, np.nan)

# 4) Outliers de porte: escolas com mais de MAX_SALAS salas
df = df[(df[COL_SALAS] > 0) & (df[COL_SALAS] <= MAX_SALAS)].copy()
log.append((f"com 1 a {MAX_SALAS} salas", len(df)))

# 5) Alvo: soma dos dispositivos dos alunos (exige as 3 colunas informadas) / salas utilizadas
df[ALVO] = df[COLS_DISPOSITIVOS].sum(axis=1, min_count=3) / df[COL_SALAS]

# 6) Remove linhas com qualquer ausente no alvo ou nas preditoras
base = df[[ALVO] + PREDITORAS].dropna().copy()
log.append(("sem valores ausentes (final)", len(base)))

# Tipos: indicadores como inteiro 0/1, contagens como inteiro, alvo com 4 casas
for c in PREDITORAS:
    base[c] = base[c].astype(int)
base[ALVO] = base[ALVO].round(4)

# Validação
assert base[PREDITORAS].isna().sum().sum() == 0
assert np.isfinite(base.to_numpy(dtype=float)).all(), "Base contém valores não finitos"
assert (base[ALVO] >= 0).all()
for c in [p for p in PREDITORAS if p.startswith("IN_")]:
    assert set(base[c].unique()) <= {0, 1}, c

base.to_csv(SAIDA, index=False, encoding="utf-8")

print("Etapas de filtragem:")
for nome, n in log:
    print(f"  {nome:35s} {n:>9,}".replace(",", "."))
print(f"\nArquivo gerado: {SAIDA}  ({base.shape[0]} linhas x {base.shape[1]} colunas)")
print("\nResumo:")
print(base.describe().T.round(3).to_string())

