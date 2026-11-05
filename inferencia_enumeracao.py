"""Inferência por enumeração (ENUMERATION-ASK) em uma rede Bayesiana discreta.

Tudo é implementado do zero: aprendizado das CPTs (contagem + Laplace) e o algoritmo de
inferência. O pandas é usado apenas para ler o CSV.

Uso:
    python inferencia_enumeracao.py dados_etapa1.csv
    python inferencia_enumeracao.py dados_etapa1.csv "infra_digital_basica | TP_LOCALIZACAO=2, IN_LABORATORIO_INFORMATICA=1"

Formato de consulta: "VARIAVEL_CONSULTA | evidencia1=valor, evidencia2=valor"
(a parte após "|" é opcional; os valores são inteiros).
"""
import sys
from dataclasses import dataclass
from itertools import product

import pandas as pd

# Estrutura do DAG: nó -> pais. A ordem das chaves DEVE ser topológica (pais antes dos filhos).
ESTRUTURA = {
    "TP_LOCALIZACAO": [],
    "TP_DEPENDENCIA": ["TP_LOCALIZACAO"],
    "IN_ENERGIA_GERADOR_FOSSIL": ["TP_LOCALIZACAO"],
    "IN_BIBLIOTECA": ["TP_DEPENDENCIA"],
    "IN_LABORATORIO_INFORMATICA": ["TP_DEPENDENCIA", "IN_ENERGIA_GERADOR_FOSSIL"],
    "infra_digital_basica": [
        "IN_ENERGIA_GERADOR_FOSSIL",
        "IN_LABORATORIO_INFORMATICA",
        "IN_BIBLIOTECA",
    ],
}


# --------------------------------------------------------------------------------------
# Rede Bayesiana e aprendizado das CPTs
# --------------------------------------------------------------------------------------
@dataclass
class RedeBayesiana:
    estrutura: dict  # nó -> lista de pais (em ordem topológica)
    dominios: dict   # nó -> lista de valores possíveis
    cpts: dict       # nó -> {(valores dos pais): {valor do nó: probabilidade}}

    def p(self, var, valor, atribuicao):
        """P(var = valor | pais(var)), com os pais lidos de 'atribuicao'."""
        cfg = tuple(atribuicao[pai] for pai in self.estrutura[var])
        return self.cpts[var][cfg][valor]


def verificar_ordem_topologica(estrutura):
    vistos = set()
    for no, pais in estrutura.items():
        for pai in pais:
            assert pai in vistos, f"Ordem não topológica: {pai} deveria vir antes de {no}"
        vistos.add(no)


def aprender_cpts(df, estrutura, alpha=1.0):
    """Estima as CPTs por contagem com suavização de Laplace:
    P(x | c) = (N(x, c) + alpha) / (N(c) + alpha * |X|)."""
    dominios = {v: sorted(df[v].unique().tolist()) for v in estrutura}
    cpts = {}
    for v, pais in estrutura.items():
        contagem = {
            cfg: {x: 0 for x in dominios[v]}
            for cfg in product(*[dominios[p] for p in pais])
        }
        for linha in df[pais + [v]].itertuples(index=False, name=None):
            contagem[tuple(linha[:-1])][linha[-1]] += 1
        cpts[v] = {}
        for cfg, c in contagem.items():
            total = sum(c.values()) + alpha * len(dominios[v])
            cpts[v][cfg] = {x: (n + alpha) / total for x, n in c.items()}
    return RedeBayesiana(estrutura, dominios, cpts)


# --------------------------------------------------------------------------------------
# Inferência por enumeração
# --------------------------------------------------------------------------------------
def enumerar_todas(variaveis, evidencia, rede):
    """ENUMERATE-ALL: soma, sobre as variáveis ocultas, do produto das probabilidades
    condicionais, percorrendo as variáveis em ordem topológica."""
    if not variaveis:
        return 1.0
    y, resto = variaveis[0], variaveis[1:]
    if y in evidencia:
        return rede.p(y, evidencia[y], evidencia) * enumerar_todas(resto, evidencia, rede)
    total = 0.0
    for valor in rede.dominios[y]:
        estendida = {**evidencia, y: valor}
        total += rede.p(y, valor, estendida) * enumerar_todas(resto, estendida, rede)
    return total


def consulta(rede, X, evidencia=None):
    """ENUMERATION-ASK: retorna P(X | evidencia) como dicionário {valor: probabilidade}."""
    evidencia = dict(evidencia or {})
    if X not in rede.estrutura:
        raise ValueError(f"Variável de consulta desconhecida: {X}")
    if X in evidencia:
        raise ValueError(f"{X} não pode ser consulta e evidência ao mesmo tempo")
    for var, val in evidencia.items():
        if var not in rede.estrutura:
            raise ValueError(f"Variável de evidência desconhecida: {var}")
        if val not in rede.dominios[var]:
            raise ValueError(f"Valor {val} fora do domínio de {var}: {rede.dominios[var]}")

    variaveis = list(rede.estrutura)  # ordem topológica
    Q = {x: enumerar_todas(variaveis, {**evidencia, X: x}, rede) for x in rede.dominios[X]}
    z = sum(Q.values())
    if z == 0:
        raise ValueError("Evidência com probabilidade zero no modelo")
    return {x: q / z for x, q in Q.items()}


# --------------------------------------------------------------------------------------
# Validação: marginalização da distribuição conjunta completa (algoritmo independente)
# --------------------------------------------------------------------------------------
def prob_conjunta(rede, atribuicao):
    """P(x1, ..., xn) = produto dos P(xi | pais(xi))."""
    prob = 1.0
    for v in rede.estrutura:
        prob *= rede.p(v, atribuicao[v], atribuicao)
    return prob


def consulta_por_conjunta(rede, X, evidencia=None):
    evidencia = evidencia or {}
    variaveis = list(rede.estrutura)
    soma = {x: 0.0 for x in rede.dominios[X]}
    for valores in product(*[rede.dominios[v] for v in variaveis]):
        a = dict(zip(variaveis, valores))
        if all(a[k] == v for k, v in evidencia.items()):
            soma[a[X]] += prob_conjunta(rede, a)
    z = sum(soma.values())
    return {x: s / z for x, s in soma.items()}


def autoteste(rede):
    variaveis = list(rede.estrutura)
    # 1) a conjunta soma 1
    total = sum(
        prob_conjunta(rede, dict(zip(variaveis, valores)))
        for valores in product(*[rede.dominios[v] for v in variaveis])
    )
    assert abs(total - 1.0) < 1e-9, f"A conjunta soma {total}, e não 1"
    # 2) enumeração == marginalização da conjunta, para todas as consultas com 0 ou 1 evidência
    maior_diferenca, n = 0.0, 0
    for X in variaveis:
        evidencias = [{}] + [
            {Z: z} for Z in variaveis if Z != X for z in rede.dominios[Z]
        ]
        for ev in evidencias:
            a, b = consulta(rede, X, ev), consulta_por_conjunta(rede, X, ev)
            maior_diferenca = max(maior_diferenca, max(abs(a[x] - b[x]) for x in a))
            n += 1
    assert maior_diferenca < 1e-12, f"Divergência de {maior_diferenca}"
    print(f"Autoteste OK: conjunta soma 1; {n} consultas conferidas contra a marginalização "
          f"da conjunta (maior diferença {maior_diferenca:.1e}).\n")


# --------------------------------------------------------------------------------------
# Utilitários de apresentação
# --------------------------------------------------------------------------------------
def freq_empirica(df, X, evidencia):
    sub = df
    for var, val in evidencia.items():
        sub = sub[sub[var] == val]
    n = len(sub)
    if n == 0:
        return None, 0
    contagem = sub[X].value_counts(normalize=True)
    return contagem.to_dict(), n


def parse_consulta(texto):
    if "|" in texto:
        X, resto = texto.split("|", 1)
        evidencia = {}
        for par in resto.split(","):
            if par.strip():
                k, v = par.split("=")
                evidencia[k.strip()] = int(v.strip())
    else:
        X, evidencia = texto, {}
    return X.strip(), evidencia


def mostrar(rede, df, X, evidencia):
    rotulo = ", ".join(f"{k}={v}" for k, v in evidencia.items()) or "—"
    print(f"P({X} | {rotulo})")
    p = consulta(rede, X, evidencia)
    emp, n = freq_empirica(df, X, evidencia)
    print("   valor   rede (enumeração)   frequência empírica")
    for x in rede.dominios[X]:
        e = f"{emp.get(x, 0.0):.4f}" if emp else "  n/d "
        print(f"   {x!s:>5}   {p[x]:>17.4f}   {e:>19}")
    print(f"   (empírico baseado em {n} escolas)\n")


CONSULTAS_EXEMPLO = [
    "infra_digital_basica",
    "infra_digital_basica | TP_LOCALIZACAO=2",
    "infra_digital_basica | TP_LOCALIZACAO=2, TP_DEPENDENCIA=3",
    "infra_digital_basica | IN_LABORATORIO_INFORMATICA=1",
    "infra_digital_basica | TP_DEPENDENCIA=3, IN_ENERGIA_GERADOR_FOSSIL=1",
    "TP_LOCALIZACAO | infra_digital_basica=1",
    "IN_ENERGIA_GERADOR_FOSSIL | infra_digital_basica=1",
    "TP_DEPENDENCIA | infra_digital_basica=1, TP_LOCALIZACAO=2",
    "IN_LABORATORIO_INFORMATICA | infra_digital_basica=0, TP_DEPENDENCIA=3",
]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    colunas = list(ESTRUTURA)
    df = pd.read_csv(sys.argv[1])[colunas].dropna().astype(int)
    verificar_ordem_topologica(ESTRUTURA)
    rede = aprender_cpts(df, ESTRUTURA, alpha=1.0)
    print(f"{len(df)} escolas | domínios: {rede.dominios}\n")
    autoteste(rede)
    consultas = sys.argv[2:] or CONSULTAS_EXEMPLO
    for texto in consultas:
        X, evidencia = parse_consulta(texto)
        mostrar(rede, df, X, evidencia)


if __name__ == "__main__":
    main()
