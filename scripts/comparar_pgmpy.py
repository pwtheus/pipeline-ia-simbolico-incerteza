"""Compara a inferência por enumeração própria com o pgmpy (VariableElimination).

Requer: pip install pgmpy   (e o arquivo inferencia_enumeracao.py na mesma pasta)

Uso:
    python comparar_pgmpy.py dados_etapa1.csv

Duas comparações:
  A) Mesmas CPTs nas duas implementações (as estimadas por inferencia_enumeracao.py):
     isola a diferença entre os algoritmos de inferência.
  B) CPTs estimadas pelo próprio pgmpy (BayesianEstimator, prior K2 = pseudo-contagem 1,
     equivalente à suavização de Laplace com alpha = 1): compara estimação + inferência.
"""
import sys
from itertools import product

import pandas as pd

from inferencia_enumeracao import (
    CONSULTAS_EXEMPLO,
    ESTRUTURA,
    aprender_cpts,
    consulta,
    parse_consulta,
    verificar_ordem_topologica,
)

TOLERANCIA = 1e-9


# --------------------------------------------------------------------------------------
# Conversão da nossa rede para o pgmpy
# --------------------------------------------------------------------------------------
def valores_cpd(rede, v):
    """Matriz (|dominio(v)| x prod(|dominio(pais)|)) no formato do TabularCPD.
    Colunas na ordem de itertools.product sobre os pais (o último pai varia mais rápido),
    que é a ordem usada pelo pgmpy."""
    pais = rede.estrutura[v]
    configs = list(product(*[rede.dominios[p] for p in pais]))
    return [[rede.cpts[v][cfg][x] for cfg in configs] for x in rede.dominios[v]]


def classe_modelo_pgmpy():
    try:
        from pgmpy.models import DiscreteBayesianNetwork as ModeloBayes  # pgmpy >= 1.0
    except ImportError:
        from pgmpy.models import BayesianNetwork as ModeloBayes  # pgmpy < 1.0
    return ModeloBayes


def estimar_modelo_pgmpy(estrutura, df):
    """Rede com as CPTs estimadas pelo pgmpy (prior K2 = pseudo-contagem 1 = Laplace alpha=1).
    Usa BayesianEstimator.get_parameters + add_cpds, em vez de modelo.fit(..., prior_type=...),
    cujos argumentos mudam entre versões do pgmpy."""
    from pgmpy.estimators import BayesianEstimator

    arestas = [(pai, filho) for filho, pais in estrutura.items() for pai in pais]
    modelo = classe_modelo_pgmpy()(arestas)
    estimador = BayesianEstimator(modelo, df)
    try:
        cpds = estimador.get_parameters(prior_type="K2")
    except TypeError:  # versões em que o prior é dado só por pseudo-contagens
        cpds = estimador.get_parameters(prior_type="dirichlet", pseudo_counts=1)
    modelo.add_cpds(*cpds)
    modelo.check_model()
    return modelo


def construir_modelo_pgmpy(rede):
    from pgmpy.factors.discrete import TabularCPD

    arestas = [(pai, filho) for filho, pais in rede.estrutura.items() for pai in pais]
    modelo = classe_modelo_pgmpy()(arestas)
    state_names = {v: list(rede.dominios[v]) for v in rede.estrutura}
    cpds = []
    for v, pais in rede.estrutura.items():
        cpds.append(
            TabularCPD(
                variable=v,
                variable_card=len(rede.dominios[v]),
                values=valores_cpd(rede, v),
                evidence=pais or None,
                evidence_card=[len(rede.dominios[p]) for p in pais] or None,
                state_names={x: state_names[x] for x in [v] + pais},
            )
        )
    modelo.add_cpds(*cpds)
    modelo.check_model()
    return modelo


def consulta_pgmpy(inferencia, X, evidencia):
    fator = inferencia.query(variables=[X], evidence=evidencia or None, show_progress=False)
    estados = fator.state_names[X]
    return {estado: float(p) for estado, p in zip(estados, fator.values)}


# --------------------------------------------------------------------------------------
# Comparações
# --------------------------------------------------------------------------------------
def comparar_inferencia(rede, inferencia, rotulo):
    print(f"=== {rotulo} ===")
    pior = 0.0
    for texto in CONSULTAS_EXEMPLO:
        X, ev = parse_consulta(texto)
        a = consulta(rede, X, ev)
        b = consulta_pgmpy(inferencia, X, ev)
        diferenca = max(abs(a[x] - b[x]) for x in a)
        pior = max(pior, diferenca)
        ev_txt = ", ".join(f"{k}={v}" for k, v in ev.items()) or "—"
        print(f"P({X} | {ev_txt})")
        for x in a:
            print(f"   {x!s:>3}: enumeração = {a[x]:.6f} | pgmpy VE = {b[x]:.6f} | |dif| = {abs(a[x]-b[x]):.1e}")
    print(f"Maior diferença absoluta: {pior:.2e}\n")
    assert pior < TOLERANCIA, f"Divergência acima da tolerância ({TOLERANCIA})"
    return pior


def comparar_cpts_estimadas(rede, modelo_pgmpy):
    """Compara as CPTs estimadas pelo pgmpy com as nossas, entrada por entrada."""
    pior = 0.0
    for v in rede.estrutura:
        cpd = modelo_pgmpy.get_cpds(v)
        ev_vars = list(cpd.variables[1:])
        matriz = cpd.get_values()
        estados_v = cpd.state_names[v]
        configs = list(product(*[cpd.state_names[e] for e in ev_vars]))
        for j, cfg in enumerate(configs):
            por_nome = dict(zip(ev_vars, cfg))
            chave = tuple(por_nome[p] for p in rede.estrutura[v])
            for i, x in enumerate(estados_v):
                pior = max(pior, abs(matriz[i][j] - rede.cpts[v][chave][x]))
    print(f"Maior diferença entre as CPTs estimadas (pgmpy K2 x Laplace alpha=1): {pior:.2e}\n")
    return pior


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    from pgmpy.inference import VariableElimination

    colunas = list(ESTRUTURA)
    df = pd.read_csv(sys.argv[1])[colunas].dropna().astype(int)
    verificar_ordem_topologica(ESTRUTURA)
    rede = aprender_cpts(df, ESTRUTURA, alpha=1.0)

    # A) mesmas CPTs, algoritmos de inferência diferentes
    modelo_a = construir_modelo_pgmpy(rede)
    comparar_inferencia(rede, VariableElimination(modelo_a), "A) Mesmas CPTs: enumeração x VariableElimination")

    # B) CPTs estimadas pelo pgmpy
    modelo_b = estimar_modelo_pgmpy(ESTRUTURA, df)
    comparar_cpts_estimadas(rede, modelo_b)
    comparar_inferencia(rede, VariableElimination(modelo_b), "B) CPTs do pgmpy: enumeração (nossas CPTs) x VariableElimination")


if __name__ == "__main__":
    main()
