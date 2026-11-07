"""Etapa 6: sistema fuzzy (Mamdani) alimentado pela rede Bayesiana da Etapa 5.

Para cada grupo de escolas com a mesma evidência, a rede Bayesiana calcula
P(infra_digital_basica | evidência) por enumeração (inferencia_enumeracao.py).
Dessa probabilidade saem as duas entradas do sistema fuzzy:

    risco     = P(infra_digital_basica = 0 | evidência)           de 0 a 1
    confianca = número de escolas parecidas no Censo (máx. 100)   de 0 a 100

A saída é a urgencia de apoio, de 0 a 100. A decisão fuzzy é comparada com a decisão
dura, que usa o limiar P(infra_digital_basica = 1 | evidência) >= 0,5.

Uso:
    python fuzzy.py dados_etapa1.csv
    python fuzzy.py dados_etapa1.csv --evidencia TP_LOCALIZACAO,TP_DEPENDENCIA,IN_ENERGIA_GERADOR_FOSSIL
    python fuzzy.py dados_etapa1.csv --grafico-pertinencia gráficos/pertinencia.png --grafico-curva gráficos/curva.png

Dependências: veja requirements.txt (o scikit-fuzzy exige também networkx e packaging, que ele não instala sozinho).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from scripts.inferencia_enumeracao import ESTRUTURA, aprender_cpts, consulta, verificar_ordem_topologica

ALVO = "infra_digital_basica"
EVIDENCIA_PADRAO = ["TP_LOCALIZACAO", "TP_DEPENDENCIA", "IN_ENERGIA_GERADOR_FOSSIL"]
LIMITE_CONFIANCA = 100   # com 100 escolas parecidas ou mais, a confiança é máxima
LIMIAR_DURO = 0.5        # decisão dura: tem infraestrutura se P(Y = 1 | evidência) >= 0,5
LIMIAR_URGENCIA = 50     # decisão fuzzy binária: priorizar se a urgência for >= 50

# Cores (paleta categórica validada, slots 1 a 3). Uma cor por nível, em todas as variáveis.
AZUL, LARANJA, VERDE = "#2a78d6", "#eb6834", "#1baf7a"
COR_DO_TERMO = {"baixo": AZUL, "baixa": AZUL, "medio": LARANJA, "media": LARANJA,
                "alto": VERDE, "alta": VERDE}
NOME_DO_TERMO = {"baixo": "baixo", "medio": "médio", "alto": "alto",
                 "baixa": "baixa", "media": "média", "alta": "alta"}


# --------------------------------------------------------------------------------------
# 1) Sistema fuzzy: variáveis linguísticas, funções de pertinência e regras
# --------------------------------------------------------------------------------------
def construir_sistema():
    """Devolve (ControlSystem, (risco, confianca, urgencia))."""
    risco = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "risco")
    confianca = ctrl.Antecedent(np.arange(0, LIMITE_CONFIANCA + 1, 1), "confianca")
    urgencia = ctrl.Consequent(np.arange(0, 101, 1), "urgencia")

    # Funções de pertinência triangulares. Em cada variável, os termos se cruzam
    # com cobertura de todo o universo; as pertinências não precisam somar 1.
    risco["baixo"] = fuzz.trimf(risco.universe, [0, 0, 0.5])
    risco["medio"] = fuzz.trimf(risco.universe, [0.25, 0.5, 0.75])
    risco["alto"] = fuzz.trimf(risco.universe, [0.5, 1, 1])

    confianca["baixa"] = fuzz.trimf(confianca.universe, [0, 0, LIMITE_CONFIANCA])
    confianca["alta"] = fuzz.trimf(confianca.universe, [0, LIMITE_CONFIANCA, LIMITE_CONFIANCA])

    urgencia["baixa"] = fuzz.trimf(urgencia.universe, [0, 0, 50])
    urgencia["media"] = fuzz.trimf(urgencia.universe, [25, 50, 75])
    urgencia["alta"] = fuzz.trimf(urgencia.universe, [50, 100, 100])

    # E = mínimo, OU = máximo (padrão do scikit-fuzzy)
    regras = [
        ctrl.Rule(risco["alto"] & confianca["alta"], urgencia["alta"]),
        ctrl.Rule(risco["medio"] | confianca["baixa"], urgencia["media"]),
        ctrl.Rule(risco["baixo"], urgencia["baixa"]),
    ]
    return ctrl.ControlSystem(regras), (risco, confianca, urgencia)


REGRAS_EM_TEXTO = [
    "R1: SE risco é ALTO E confiança é ALTA ENTÃO urgência é ALTA",
    "R2: SE risco é MÉDIO OU confiança é BAIXA ENTÃO urgência é MÉDIA",
    "R3: SE risco é BAIXO ENTÃO urgência é BAIXA",
]


def calcular_urgencia(simulacao, risco, confianca):
    """Roda o sistema Mamdani para uma entrada e devolve a urgência (0 a 100)."""
    simulacao.input["risco"] = float(risco)
    simulacao.input["confianca"] = float(min(confianca, LIMITE_CONFIANCA))
    simulacao.compute()
    return float(simulacao.output["urgencia"])


def rotulo_urgencia(valor, urgencia):
    """Termo linguístico com maior pertinência no valor defuzzificado."""
    pertinencias = {
        termo: float(fuzz.interp_membership(urgencia.universe, urgencia[termo].mf, valor))
        for termo in ("baixa", "media", "alta")
    }
    return max(pertinencias, key=pertinencias.get)


# --------------------------------------------------------------------------------------
# 2) Ponte com a Etapa 5: probabilidades da rede Bayesiana viram entradas fuzzy
# --------------------------------------------------------------------------------------
def montar_rede(df, alpha=1.0):
    verificar_ordem_topologica(ESTRUTURA)
    return aprender_cpts(df, ESTRUTURA, alpha=alpha)


def tabela_por_evidencia(df_alvo, df_treino, rede, evidencia_vars, sistema, variaveis):
    """Uma linha por configuração da evidência presente em df_alvo.

    A rede e a contagem de escolas parecidas (N) vêm sempre do df_treino. Assim, a mesma função
    serve para a tabela explicativa (df_alvo = df_treino) e para a validação cruzada
    (df_alvo = fold de teste)."""
    _, _, urgencia = variaveis
    simulacao = ctrl.ControlSystemSimulation(sistema)
    n_treino = df_treino.groupby(evidencia_vars).size()
    linhas = []
    for cfg, grupo in df_alvo.groupby(evidencia_vars):
        cfg = cfg if isinstance(cfg, tuple) else (cfg,)
        evidencia = {v: int(x) for v, x in zip(evidencia_vars, cfg)}
        p = consulta(rede, ALVO, evidencia)            # inferência por enumeração (Etapa 5)
        n = int(n_treino.get(cfg if len(cfg) > 1 else cfg[0], 0))
        risco = p[0]
        u = calcular_urgencia(simulacao, risco, n)
        linhas.append({
            **evidencia,
            "N": n,
            "P_infra": p[1],
            "risco": risco,
            "confianca": min(n, LIMITE_CONFIANCA),
            "urgencia": u,
            "rotulo_fuzzy": rotulo_urgencia(u, urgencia),
            "real_infra": grupo[ALVO].mean(),
        })
    tabela = pd.DataFrame(linhas)
    tabela["decisao_dura"] = np.where(tabela["P_infra"] >= LIMIAR_DURO, "tem infra", "sem infra")
    tabela["decisao_fuzzy"] = np.where(tabela["urgencia"] >= LIMIAR_URGENCIA, "sem infra", "tem infra")
    return tabela


def validacao_cruzada(df, evidencia_vars, sistema, variaveis, k=5, seed=42):
    """5-fold estratificado (mesmo protocolo da Etapa 1). Em cada fold, as CPTs e as contagens
    vêm só do treino; as decisões são tomadas nas escolas do teste. Devolve uma linha por escola."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    colunas = ["P_infra", "risco", "N", "urgencia", "rotulo_fuzzy", "decisao_dura", "decisao_fuzzy"]
    partes = []
    for treino_idx, teste_idx in skf.split(df, df[ALVO]):
        treino, teste = df.iloc[treino_idx], df.iloc[teste_idx]
        rede = montar_rede(treino)
        tabela = tabela_por_evidencia(teste, treino, rede, evidencia_vars, sistema, variaveis)
        partes.append(teste.merge(tabela[evidencia_vars + colunas], on=evidencia_vars, how="left"))
    return pd.concat(partes, ignore_index=True)


def comparar_decisoes(escolas):
    """Compara decisão dura x decisão fuzzy com o rótulo real (use as escolas do teste)."""
    real_sem = (escolas[ALVO] == 0).to_numpy()
    resumo = {}
    for nome, coluna in [("dura", "decisao_dura"), ("fuzzy", "decisao_fuzzy")]:
        prevista_sem = (escolas[coluna] == "sem infra").to_numpy()
        resumo[nome] = {
            "acuracia": float((prevista_sem == real_sem).mean()),
            "recall_sem_infra": float((prevista_sem & real_sem).sum() / real_sem.sum()),
            "recall_com_infra": float((~prevista_sem & ~real_sem).sum() / (~real_sem).sum()),
        }
    # AUC: quão bem cada saída ordena as escolas sem infraestrutura acima das que têm.
    resumo["auc"] = {
        "decisao dura (sim/não)": float(roc_auc_score(real_sem, escolas["decisao_dura"] == "sem infra")),
        "probabilidade da rede": float(roc_auc_score(real_sem, escolas["risco"])),
        "urgência fuzzy": float(roc_auc_score(real_sem, escolas["urgencia"])),
    }
    resumo["concordancia"] = float((escolas["decisao_dura"] == escolas["decisao_fuzzy"]).mean())
    por_rotulo = (escolas.groupby("rotulo_fuzzy")[ALVO]
                  .agg(escolas="size", pct_com_infra_real="mean")
                  .reindex(["baixa", "media", "alta"]))
    por_dura = (escolas.groupby("decisao_dura")[ALVO]
                .agg(escolas="size", pct_com_infra_real="mean"))
    return resumo, por_rotulo, por_dura


# --------------------------------------------------------------------------------------
# 3) Gráficos
# --------------------------------------------------------------------------------------
def _estilo(ax):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)


def grafico_pertinencia(variaveis, caminho=None):
    """Funções de pertinência das 3 variáveis linguísticas."""
    import matplotlib.pyplot as plt
    risco, confianca, urgencia = variaveis
    titulos = {"risco": "Risco (P de NÃO ter infraestrutura)",
               "confianca": "Confiança (escolas parecidas)",
               "urgencia": "Urgência de apoio (saída)"}
    fig, eixos = plt.subplots(1, 3, figsize=(13, 3.4), dpi=130)
    for ax, var in zip(eixos, (risco, confianca, urgencia)):
        for termo in var.terms:
            ax.plot(var.universe, var[termo].mf, color=COR_DO_TERMO[termo], linewidth=2,
                    label=NOME_DO_TERMO[termo])
        ax.set_title(titulos[var.label], fontsize=10, loc="left")
        ax.set_ylim(-0.03, 1.3)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_ylabel("pertinência")
        ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=3)
        _estilo(ax)
    fig.tight_layout()
    if caminho:
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(caminho, bbox_inches="tight")
    return fig


def grafico_curva(sistema, caminho=None):
    """Urgência fuzzy x decisão dura, variando a probabilidade da rede de 0 a 1."""
    import matplotlib.pyplot as plt
    simulacao = ctrl.ControlSystemSimulation(sistema)
    p = np.linspace(0, 1, 101)
    curvas = {}
    for n in (10, LIMITE_CONFIANCA):
        curvas[n] = [calcular_urgencia(simulacao, 1 - x, n) for x in p]
    fig, ax = plt.subplots(figsize=(7.2, 4), dpi=130)
    ax.step(p, np.where(p >= LIMIAR_DURO, 0, 100), where="post", color="#52514e",
            linewidth=1.8, linestyle="--", label="decisão dura (limiar 0,5): 100 ou 0")
    ax.plot(p, curvas[LIMITE_CONFIANCA], color=AZUL, linewidth=2.2,
            label=f"fuzzy, {LIMITE_CONFIANCA} escolas parecidas")
    ax.plot(p, curvas[10], color=LARANJA, linewidth=2.2, label="fuzzy, 10 escolas parecidas")
    ax.axhline(LIMIAR_URGENCIA, color="#b9b8b2", linewidth=0.8)
    ax.set_xlabel("P(infra_digital_basica = 1 | evidência), vinda da rede Bayesiana")
    ax.set_ylabel("urgência de apoio (0 a 100)")
    ax.set_ylim(-3, 103)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _estilo(ax)
    fig.tight_layout()
    if caminho:
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(caminho, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------------------
# 4) Execução pela linha de comando
# --------------------------------------------------------------------------------------
def executar(caminho_csv, evidencia_vars):
    """Lê os dados, monta a rede e o sistema fuzzy e devolve também a tabela por configuração."""
    colunas = list(ESTRUTURA)
    df = pd.read_csv(caminho_csv)[colunas].dropna().astype(int)
    rede = montar_rede(df)
    sistema, variaveis = construir_sistema()
    tabela = tabela_por_evidencia(df, df, rede, evidencia_vars, sistema, variaveis)
    return df, rede, sistema, variaveis, tabela


def main():
    ap = argparse.ArgumentParser(description="Etapa 6: sistema fuzzy Mamdani sobre a rede Bayesiana.")
    ap.add_argument("csv")
    ap.add_argument("--evidencia", default=",".join(EVIDENCIA_PADRAO),
                    help="variáveis observadas, separadas por vírgula")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grafico-pertinencia")
    ap.add_argument("--grafico-curva")
    args = ap.parse_args()

    evidencia_vars = [v.strip() for v in args.evidencia.split(",") if v.strip()]
    invalidas = [v for v in evidencia_vars if v not in ESTRUTURA or v == ALVO]
    if invalidas:
        raise SystemExit(f"Evidência inválida: {invalidas}. Use nós da rede, exceto {ALVO}.")

    df, rede, sistema, variaveis, tabela = executar(args.csv, evidencia_vars)
    print(f"{len(df)} escolas | evidência: {evidencia_vars}\n")
    print("Regras:")
    for r in REGRAS_EM_TEXTO:
        print("  " + r)
    print("\nUma linha por configuração da evidência (CPTs estimadas com todas as escolas):")
    cols = evidencia_vars + ["N", "P_infra", "risco", "urgencia", "rotulo_fuzzy",
                             "decisao_dura", "decisao_fuzzy", "real_infra"]
    print(tabela[cols].round(3).to_string(index=False))

    escolas = validacao_cruzada(df, evidencia_vars, sistema, variaveis, args.k, args.seed)
    resumo, por_rotulo, por_dura = comparar_decisoes(escolas)
    print(f"\nValidação cruzada {args.k}-fold (seed {args.seed}), decisões tomadas só no fold de teste:")
    print(f"  Concordância entre decisão dura e fuzzy (sim/não): {resumo['concordancia']:.2%}")
    for nome in ("dura", "fuzzy"):
        r = resumo[nome]
        print(f"  {nome:5}: acurácia {r['acuracia']:.2%} | recall sem infra {r['recall_sem_infra']:.2%} "
              f"| recall com infra {r['recall_com_infra']:.2%}")
    print("  AUC (ordenação das escolas sem infraestrutura):")
    for nome, v in resumo["auc"].items():
        print(f"    {nome:26} {v:.3f}")
    print("\nEscolas por rótulo fuzzy (e % que realmente têm infraestrutura):")
    print(por_rotulo.round(3).to_string())
    print("\nEscolas por decisão dura:")
    print(por_dura.round(3).to_string())

    if args.grafico_pertinencia:
        grafico_pertinencia(variaveis, args.grafico_pertinencia)
        print(f"\nGráfico salvo em {args.grafico_pertinencia}")
    if args.grafico_curva:
        grafico_curva(sistema, args.grafico_curva)
        print(f"Gráfico salvo em {args.grafico_curva}")


if __name__ == "__main__":
    main()

