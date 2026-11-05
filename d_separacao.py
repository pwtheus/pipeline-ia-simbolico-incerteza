"""d-separação (implementação própria) + teste empírico de independência condicional.

Uso:
    python d_separacao.py dados_etapa1.csv

Para cada independência X ⟂ Y | Z:
  1) lista todos os caminhos entre X e Y no DAG e diz, para cada um, se está bloqueado e por quê;
  2) confere o veredito com um segundo algoritmo (grafo ancestral moralizado);
  3) testa a independência nos dados: qui-quadrado estratificado por Z (soma dos estratos) e
     informação mútua condicional I(X;Y|Z), em bits.
Requer: pandas, numpy, scipy (só para o p-valor do qui-quadrado).
"""
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist

from inferencia_enumeracao import ESTRUTURA

ALIAS = {
    "TP_LOCALIZACAO": "LOC",
    "TP_DEPENDENCIA": "DEP",
    "IN_ENERGIA_GERADOR_FOSSIL": "GER",
    "IN_BIBLIOTECA": "BIB",
    "IN_LABORATORIO_INFORMATICA": "LAB",
    "infra_digital_basica": "Y",
}
POR_ALIAS = {a: n for n, a in ALIAS.items()}


# --------------------------------------------------------------------------------------
# d-separação
# --------------------------------------------------------------------------------------
def descendentes(estrutura, no):
    filhos = {n: [f for f, ps in estrutura.items() if n in ps] for n in estrutura}
    achados, pilha = set(), [no]
    while pilha:
        for f in filhos[pilha.pop()]:
            if f not in achados:
                achados.add(f)
                pilha.append(f)
    return achados


def caminhos_simples(estrutura, X, Y):
    """Todos os caminhos sem repetir nó entre X e Y, ignorando o sentido das arestas."""
    vizinhos = {n: set(ps) | {f for f, p2 in estrutura.items() if n in p2} for n, ps in estrutura.items()}
    saida = []

    def dfs(atual, caminho):
        if atual == Y:
            saida.append(list(caminho))
            return
        for v in sorted(vizinhos[atual]):
            if v not in caminho:
                dfs(v, caminho + [v])

    dfs(X, [X])
    return saida


def analisar_caminho(estrutura, caminho, Z):
    """Retorna (texto do caminho, bloqueado?, motivo)."""
    Z = set(Z)
    texto = ALIAS[caminho[0]]
    for u, v in zip(caminho, caminho[1:]):
        texto += (" → " if u in estrutura[v] else " ← ") + ALIAS[v]
    for i in range(1, len(caminho) - 1):
        ant, no, prox = caminho[i - 1], caminho[i], caminho[i + 1]
        colisor = ant in estrutura[no] and prox in estrutura[no]
        if colisor:
            if no not in Z and not (descendentes(estrutura, no) & Z):
                return texto, True, f"colisor em {ALIAS[no]} (nem ele nem descendentes observados)"
        elif no in Z:
            return texto, True, f"{ALIAS[no]} observado bloqueia (cadeia/bifurcação)"
    abertos = [ALIAS[caminho[i]] for i in range(1, len(caminho) - 1)
               if caminho[i - 1] in estrutura[caminho[i]] and caminho[i + 1] in estrutura[caminho[i]]]
    motivo = "aberto" + (f" (colisor {', '.join(abertos)} ativado pela evidência)" if abertos else "")
    return texto, False, motivo


def d_separados_por_caminhos(estrutura, X, Y, Z):
    return all(analisar_caminho(estrutura, c, Z)[1] for c in caminhos_simples(estrutura, X, Y))


def d_separados_moral(estrutura, X, Y, Z):
    """Segundo algoritmo: grafo ancestral de {X,Y}∪Z, moralizado; remove Z e testa conexão."""
    alvo = {X, Y} | set(Z)
    relevantes, pilha = set(alvo), list(alvo)
    while pilha:
        for p in estrutura[pilha.pop()]:
            if p not in relevantes:
                relevantes.add(p)
                pilha.append(p)
    adj = {n: set() for n in relevantes}
    for filho in relevantes:
        pais = estrutura[filho]
        for p in pais:
            adj[p].add(filho)
            adj[filho].add(p)
        for a, b in combinations(pais, 2):  # "casar" os pais
            adj[a].add(b)
            adj[b].add(a)
    visitados, fila = {X}, [X]
    while fila:
        for m in adj[fila.pop()]:
            if m in Z or m in visitados:
                continue
            if m == Y:
                return False
            visitados.add(m)
            fila.append(m)
    return True


# --------------------------------------------------------------------------------------
# Teste empírico de independência condicional
# --------------------------------------------------------------------------------------
def teste_empirico(df, X, Y, Z):
    n = len(df)
    qui2, gl, cmi_nats = 0.0, 0, 0.0
    estratos_ignorados = 0
    obs_esparsas = 0
    grupos = df.groupby(list(Z)) if Z else [((), df)]
    for _, sub in grupos:
        tab = pd.crosstab(sub[X], sub[Y]).values.astype(float)
        r, c = tab.shape
        if r < 2 or c < 2:
            estratos_ignorados += 1
            continue
        ns = tab.sum()
        lin, col = tab.sum(axis=1, keepdims=True), tab.sum(axis=0, keepdims=True)
        esperado = lin @ col / ns
        qui2 += ((tab - esperado) ** 2 / esperado).sum()
        gl += (r - 1) * (c - 1)
        mascara = tab > 0
        cmi_nats += (tab[mascara] * np.log((tab * ns / (lin @ col))[mascara])).sum() / n
        if esperado.min() < 5:
            obs_esparsas += int(ns)
    p = float(chi2_dist.sf(qui2, gl)) if gl > 0 else float("nan")
    return {
        "qui2": qui2,
        "gl": gl,
        "p": p,
        "cmi_bits": cmi_nats / np.log(2),
        "obs_esparsas": obs_esparsas,
        "estratos_ignorados": estratos_ignorados,
    }


# --------------------------------------------------------------------------------------
# Relatório
# --------------------------------------------------------------------------------------
def relatorio(df, X, Y, Z, rotulo):
    print("-" * 78)
    conj = ", ".join(ALIAS[z] for z in Z) if Z else "∅"
    print(f"{rotulo}:  {ALIAS[X]} ⟂ {ALIAS[Y]} | {{{conj}}}")
    print("-" * 78)
    for caminho in caminhos_simples(ESTRUTURA, X, Y):
        texto, bloqueado, motivo = analisar_caminho(ESTRUTURA, caminho, Z)
        print(f"  {'bloqueado' if bloqueado else 'ABERTO   '}  {texto:<34} {motivo}")
    a = d_separados_por_caminhos(ESTRUTURA, X, Y, Z)
    b = d_separados_moral(ESTRUTURA, X, Y, set(Z))
    assert a == b, "Os dois algoritmos de d-separação divergem"
    veredito = "d-SEPARADOS (o DAG implica independência)" if a else "d-CONECTADOS (o DAG NÃO implica independência)"
    print(f"  => {veredito}   [conferido por 2 algoritmos]")
    t = teste_empirico(df, X, Y, Z)
    print(f"  Dados: qui² = {t['qui2']:.2f}, gl = {t['gl']}, p = {t['p']:.3g}, "
          f"I(X;Y|Z) = {t['cmi_bits']:.5f} bits")
    if t["obs_esparsas"]:
        print(f"  Aviso: {t['obs_esparsas']} escolas estão em estratos com frequência esperada < 5 "
              f"(qui-quadrado menos confiável).")
    print()


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cols = list(ESTRUTURA)
    df = pd.read_csv(sys.argv[1])[cols].dropna().astype(int)
    print(f"{len(df)} escolas. Legenda: " + ", ".join(f"{a}={n}" for n, a in ALIAS.items()) + "\n")

    P = POR_ALIAS
    print("=== INDEPENDÊNCIA 1: BIBLIOTECA ⟂ LABORATÓRIO | DEPENDÊNCIA ===\n")
    relatorio(df, P["BIB"], P["LAB"], [], "Controle (sem condicionar)")
    relatorio(df, P["BIB"], P["LAB"], [P["DEP"]], "Independência 1")
    relatorio(df, P["BIB"], P["LAB"], [P["DEP"], P["Y"]], "Contraste: condicionando também no colisor Y")

    print("=== INDEPENDÊNCIA 2: DEPENDÊNCIA ⟂ GERADOR | LOCALIZAÇÃO ===\n")
    relatorio(df, P["DEP"], P["GER"], [], "Controle (sem condicionar)")
    relatorio(df, P["DEP"], P["GER"], [P["LOC"]], "Independência 2")
    relatorio(df, P["DEP"], P["GER"], [P["LOC"], P["LAB"]], "Contraste: condicionando também no colisor LAB")

    print("=== DIAGNÓSTICO EXTRA: ALVO ⟂ LOCALIZAÇÃO | {GERADOR, LAB, BIBLIOTECA} ===\n")
    relatorio(df, P["Y"], P["LOC"], [P["GER"], P["LAB"], P["BIB"]], "Diagnóstico")


if __name__ == "__main__":
    main()
