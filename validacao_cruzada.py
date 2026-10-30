"""
Validação cruzada 5-fold (estratificada) do ID3 próprio vs DecisionTreeClassifier.

Uso:
    python validacao_cruzada.py dados_etapa1.csv
"""
import argparse

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

import learn_decision_tree as ldt
from comparar_sklearn import folhas, um_hot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--alvo", default="infra_digital_basica")
    ap.add_argument("--ignorar", nargs="*", default=["CO_ENTIDADE"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dados, attrs = ldt.ler_csv(args.csv, args.alvo, args.ignorar)
    y = np.array([int(d[args.alvo]) for d in dados])
    X, _ = um_hot(dados, attrs)  # one-hot para o sklearn (colunas fixas p/ todos os folds)
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)

    for md in (3, None):
        print("=" * 66)
        print(f"{args.k}-fold estratificado | max_depth={md}")
        print("=" * 66)
        res = {"ID3 próprio": [], "sklearn (entropy)": [], "sklearn (gini)": [],
               "baseline (classe majoritária)": []}
        folhas_p, folhas_s = [], []
        print(f"{'fold':>4} {'ID3 próprio':>12} {'sk-entropy':>11} {'sk-gini':>9} {'baseline':>9}")
        for i, (tr, te) in enumerate(skf.split(X, y), 1):
            # ID3 próprio: usa os dicionários originais; o domínio vem do treino do fold
            d_tr = [dados[j] for j in tr]
            d_te = [dados[j] for j in te]
            arv = ldt.treinar(d_tr, attrs, args.alvo, md)
            acc_p = np.mean([int(ldt.prever(arv, e)) == int(e[args.alvo]) for e in d_te])
            folhas_p.append(folhas(arv))

            accs = {}
            for crit in ("entropy", "gini"):
                clf = DecisionTreeClassifier(criterion=crit, max_depth=md, random_state=0)
                clf.fit(X[tr], y[tr])
                accs[crit] = clf.score(X[te], y[te])
                if crit == "entropy":
                    folhas_s.append(clf.get_n_leaves())

            maj = np.bincount(y[tr]).argmax()
            acc_b = np.mean(y[te] == maj)

            res["ID3 próprio"].append(acc_p)
            res["sklearn (entropy)"].append(accs["entropy"])
            res["sklearn (gini)"].append(accs["gini"])
            res["baseline (classe majoritária)"].append(acc_b)
            print(f"{i:>4} {acc_p:>12.4f} {accs['entropy']:>11.4f} {accs['gini']:>9.4f} {acc_b:>9.4f}")

        print("\nAcurácia média ± desvio padrão (entre os folds):")
        for nome, v in res.items():
            print(f"  {nome:32} {np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}")
        print(f"\nFolhas médias: ID3 próprio = {np.mean(folhas_p):.1f} | "
              f"sklearn = {np.mean(folhas_s):.1f}\n")


if __name__ == "__main__":
    main()
