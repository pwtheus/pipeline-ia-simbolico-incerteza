"""
Compara o LEARN-DECISION-TREE implementado do zero com o
DecisionTreeClassifier do scikit-learn.

Uso:
    python comparar_sklearn.py dados_etapa1.csv
"""
import argparse
import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

import learn_decision_tree as ldt


# --- métricas de tamanho da árvore própria ---------------------------------
def folhas(no):
    return 1 if no.eh_folha else sum(folhas(f) for f in no.filhos.values())


def profundidade(no):
    return 0 if no.eh_folha else 1 + max(profundidade(f) for f in no.filhos.values())


def um_hot(linhas, atributos, categorias=None):
    """One-hot manual: uma coluna binária por (atributo, valor)."""
    if categorias is None:
        categorias = [(a, v) for a in atributos for v in sorted({r[a] for r in linhas})]
    X = np.array([[1 if r[a] == v else 0 for a, v in categorias] for r in linhas])
    return X, categorias


def avaliar(nome, ajustar, prever, Xtr, ytr, Xte, yte):
    t0 = time.perf_counter()
    modelo = ajustar()
    dt = time.perf_counter() - t0
    ptr, pte = prever(modelo, Xtr), prever(modelo, Xte)
    return modelo, dt, ptr, pte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--alvo", default="infra_digital_basica")
    ap.add_argument("--ignorar", nargs="*", default=["CO_ENTIDADE"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dados, attrs = ldt.ler_csv(args.csv, args.alvo, args.ignorar)
    y_all = [d[args.alvo] for d in dados]
    tr, te = train_test_split(dados, test_size=0.3, random_state=args.seed, stratify=y_all)
    ytr = np.array([int(d[args.alvo]) for d in tr])
    yte = np.array([int(d[args.alvo]) for d in te])
    Xtr, cats = um_hot(tr, attrs)
    Xte, _ = um_hot(te, attrs, cats)
    nomes = [f"{a}={v}" for a, v in cats]

    print(f"Treino: {len(tr)} | Teste: {len(te)}\n")

    for md in (3, None):
        rotulo = f"max_depth={md}"
        print("=" * 72)
        print(rotulo)
        print("=" * 72)
        linhas = []

        # --- implementação própria (ID3 multiway, entropia) ---
        t0 = time.perf_counter()
        hist = []
        arv = ldt.treinar(tr, attrs, args.alvo, md, hist)
        dt = time.perf_counter() - t0
        p_tr = np.array([int(ldt.prever(arv, e)) for e in tr])
        p_te = np.array([int(ldt.prever(arv, e)) for e in te])
        linhas.append(("ID3 próprio (multiway)", arv.atributo, profundidade(arv),
                       folhas(arv), accuracy_score(ytr, p_tr), accuracy_score(yte, p_te), dt))
        pred_proprio = p_te

        # --- scikit-learn: entropia e gini, ambos com one-hot ---
        preds = {}
        for crit in ("entropy", "gini"):
            t0 = time.perf_counter()
            clf = DecisionTreeClassifier(criterion=crit, max_depth=md, random_state=0).fit(Xtr, ytr)
            dt = time.perf_counter() - t0
            raiz = nomes[clf.tree_.feature[0]]
            linhas.append((f"sklearn ({crit}, one-hot)", raiz, clf.get_depth(),
                           clf.get_n_leaves(), clf.score(Xtr, ytr), clf.score(Xte, yte), dt))
            preds[crit] = clf.predict(Xte)
            if md == 3 and crit == "entropy":
                clf_txt = export_text(clf, feature_names=nomes)

        print(f"{'modelo':28} {'raiz':30} {'prof':>4} {'folhas':>6} {'acc_tr':>7} {'acc_te':>7} {'tempo(s)':>9}")
        for n, r, p, f, a1, a2, t in linhas:
            print(f"{n:28} {str(r):30} {p:>4} {f:>6} {a1:>7.4f} {a2:>7.4f} {t:>9.4f}")

        print("\nConcordância das previsões no teste:")
        print(f"  próprio x sklearn-entropy: {np.mean(pred_proprio == preds['entropy']):.4f}")
        print(f"  próprio x sklearn-gini   : {np.mean(pred_proprio == preds['gini']):.4f}")
        print(f"  sklearn-entropy x gini   : {np.mean(preds['entropy'] == preds['gini']):.4f}\n")

        if md == 3:
            print("Árvore do sklearn (entropy, max_depth=3):")
            print(clf_txt)
            print("Árvore própria (max_depth=3):")
            ldt.imprimir(arv)
            print()

    # --- Dados completos (mesma base do cálculo manual do documento) -------
    print("=" * 72)
    print("Dados completos (n = %d), max_depth=3: confere com o cálculo manual" % len(dados))
    print("=" * 72)
    hist = []
    arv = ldt.treinar(dados, attrs, args.alvo, 3, hist)
    print(f"H(raiz) = {hist[0]['H']:.4f}")
    for h in hist:
        if h["profundidade"] <= 3:
            gs = sorted(h["ganhos"].items(), key=lambda kv: -kv[1])
            print(f"  nivel {h['profundidade']} n={h['n']:>4} {h['contagem']} -> {h['escolhido']}"
                  f" (GI={h['ganhos'][h['escolhido']]:.4f}) | " +
                  "; ".join(f"{a}={g:.4f}" for a, g in gs[1:]))
    yfull = np.array([int(d[args.alvo]) for d in dados])
    Xfull, _ = um_hot(dados, attrs, cats)
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0).fit(Xfull, yfull)
    p_prop = np.array([int(ldt.prever(arv, e)) for e in dados])
    print(f"\nacc treino  próprio={accuracy_score(yfull, p_prop):.4f}  "
          f"sklearn-entropy={clf.score(Xfull, yfull):.4f}")
    print(f"concordância próprio x sklearn: {np.mean(p_prop == clf.predict(Xfull)):.4f}")
    print(export_text(clf, feature_names=nomes))


if __name__ == "__main__":
    main()
