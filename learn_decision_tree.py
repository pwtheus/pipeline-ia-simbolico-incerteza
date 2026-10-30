"""
LEARN-DECISION-TREE (ID3, divisão multiway) implementado do zero.
Sem pandas / sklearn: apenas biblioteca padrão do Python.

Uso:
    python learn_decision_tree.py dados_etapa1.csv
    python learn_decision_tree.py dados_etapa1.csv --max-depth 3
"""
import argparse
import csv
import math
from collections import Counter


# ----------------------------------------------------------------------------
# 1. Medidas de impureza
# ----------------------------------------------------------------------------
def entropia(exemplos, alvo):
    """H(S) = -sum_i p_i log2 p_i"""
    n = len(exemplos)
    if n == 0:
        return 0.0
    contagem = Counter(e[alvo] for e in exemplos)
    return -sum((c / n) * math.log2(c / n) for c in contagem.values())


def ganho_informacao(exemplos, atributo, alvo):
    """GI(S, A) = H(S) - sum_v |S_v|/|S| * H(S_v)"""
    n = len(exemplos)
    grupos = {}
    for e in exemplos:
        grupos.setdefault(e[atributo], []).append(e)
    resto = sum(len(g) / n * entropia(g, alvo) for g in grupos.values())
    return entropia(exemplos, alvo) - resto


def valor_plural(exemplos, alvo):
    """PLURALITY-VALUE: classe mais frequente (empate -> menor valor, determinístico)."""
    contagem = Counter(e[alvo] for e in exemplos)
    maior = max(contagem.values())
    return min(c for c, k in contagem.items() if k == maior)


# ----------------------------------------------------------------------------
# 2. Estrutura da árvore
# ----------------------------------------------------------------------------
class No:
    """Nó interno (atributo + filhos) ou folha (classe)."""

    def __init__(self, atributo=None, filhos=None, classe=None, n=0, contagem=None):
        self.atributo = atributo      # None => folha
        self.filhos = filhos or {}    # valor do atributo -> No
        self.classe = classe          # classe prevista (folha) ou plural (nó interno)
        self.n = n                    # nº de exemplos que chegaram ao nó
        self.contagem = contagem or {}

    @property
    def eh_folha(self):
        return self.atributo is None


# ----------------------------------------------------------------------------
# 3. LEARN-DECISION-TREE
# ----------------------------------------------------------------------------
def learn_decision_tree(exemplos, atributos, exemplos_pai, alvo, dominio,
                        profundidade=0, max_depth=None, historico=None):
    """
    exemplos      : lista de dicts (exemplos que chegaram a este nó)
    atributos     : lista de atributos ainda disponíveis
    exemplos_pai  : exemplos do nó pai (usado quando 'exemplos' é vazio)
    dominio       : {atributo: conjunto de TODOS os valores possíveis no dataset}
    historico     : lista opcional; registra o ganho de cada atributo em cada nó
    """
    # Caso 1: sem exemplos -> classe plural do pai
    if not exemplos:
        return No(classe=valor_plural(exemplos_pai, alvo), n=0)

    contagem = dict(Counter(e[alvo] for e in exemplos))

    # Caso 2: todos os exemplos têm a mesma classe -> folha pura
    if len(contagem) == 1:
        return No(classe=next(iter(contagem)), n=len(exemplos), contagem=contagem)

    # Caso 3: sem atributos (ou profundidade máxima) -> classe plural
    if not atributos or (max_depth is not None and profundidade >= max_depth):
        return No(classe=valor_plural(exemplos, alvo), n=len(exemplos), contagem=contagem)

    # Caso 4: escolhe o atributo de maior ganho (IMPORTANCE)
    ganhos = {a: ganho_informacao(exemplos, a, alvo) for a in atributos}
    melhor = max(atributos, key=lambda a: ganhos[a])  # empate -> primeiro da lista

    if historico is not None:
        historico.append({"profundidade": profundidade + 1, "n": len(exemplos),
                          "contagem": contagem, "H": entropia(exemplos, alvo),
                          "escolhido": melhor, "ganhos": ganhos})

    no = No(atributo=melhor, classe=valor_plural(exemplos, alvo),
            n=len(exemplos), contagem=contagem)
    restantes = [a for a in atributos if a != melhor]

    # Um ramo para CADA valor possível do atributo (multiway)
    for v in sorted(dominio[melhor]):
        sub = [e for e in exemplos if e[melhor] == v]
        no.filhos[v] = learn_decision_tree(sub, restantes, exemplos, alvo, dominio,
                                           profundidade + 1, max_depth, historico)
    return no


def treinar(exemplos, atributos, alvo, max_depth=None, historico=None):
    dominio = {a: {e[a] for e in exemplos} for a in atributos}
    return learn_decision_tree(exemplos, list(atributos), exemplos, alvo, dominio,
                               0, max_depth, historico)


# ----------------------------------------------------------------------------
# 4. Predição, avaliação e impressão
# ----------------------------------------------------------------------------
def prever(no, exemplo):
    while not no.eh_folha:
        proximo = no.filhos.get(exemplo[no.atributo])
        if proximo is None:        # valor nunca visto no treino
            return no.classe
        no = proximo
    return no.classe


def acuracia(no, exemplos, alvo):
    return sum(prever(no, e) == e[alvo] for e in exemplos) / len(exemplos)


def imprimir(no, recuo=""):
    if no.eh_folha:
        print(f"{recuo}-> classe {no.classe}  (n={no.n}, {no.contagem})")
        return
    print(f"{recuo}[{no.atributo}]  n={no.n}, {no.contagem}")
    for v, filho in no.filhos.items():
        print(f"{recuo}  {no.atributo} = {v}:")
        imprimir(filho, recuo + "    ")


# ----------------------------------------------------------------------------
# 5. Leitura do CSV e execução
# ----------------------------------------------------------------------------
def ler_csv(caminho, alvo, ignorar):
    with open(caminho, newline="", encoding="utf-8-sig") as f:
        linhas = list(csv.DictReader(f))
    atributos = [c for c in linhas[0] if c != alvo and c not in ignorar]
    return linhas, atributos


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--alvo", default="infra_digital_basica")
    ap.add_argument("--ignorar", nargs="*", default=["CO_ENTIDADE"])
    ap.add_argument("--max-depth", type=int, default=None)
    args = ap.parse_args()

    dados, attrs = ler_csv(args.csv, args.alvo, args.ignorar)
    hist = []
    arvore = treinar(dados, attrs, args.alvo, args.max_depth, hist)

    print(f"H(raiz) = {hist[0]['H']:.4f}")
    for a, g in sorted(hist[0]["ganhos"].items(), key=lambda kv: -kv[1]):
        print(f"  GI({a}) = {g:.4f}")
    print()
    imprimir(arvore)
    print(f"\nAcurácia no treino: {acuracia(arvore, dados, args.alvo):.4f}")
