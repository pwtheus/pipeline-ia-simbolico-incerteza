#!/usr/bin/env python3
"""
Regressão logística do zero (sigmoide + gradiente descendente) para a Etapa 1:
prever infra_digital_basica (0/1) a partir de dados_etapa1.csv.

Modelo:      ŷ = σ(Xθ),   σ(z) = 1 / (1 + e^-z)
Custo:       J(θ) = -(1/m) Σ [ y·log(ŷ) + (1-y)·log(1-ŷ) ]      (entropia cruzada)
Gradiente:   ∇J = (1/m) Xᵀ (σ(Xθ) - y)
Atualização: θ ← θ - α ∇J
Decisão:     classe 1 se ŷ ≥ limiar (padrão 0,5)

Pré-processamento:
  - TP_LOCALIZACAO (1 urbana, 2 rural)  -> RURAL (0/1)
  - TP_DEPENDENCIA (1 federal, 2 estadual, 3 municipal, 4 privada) -> dummies ESTADUAL, FEDERAL, PRIVADA
    (referência = municipal, a categoria mais frequente)
  - CO_ENTIDADE (identificador) é descartado.

Uso:
    python3 regressao_logistica.py dados_etapa1.csv [--alpha 1.0] [--iter 100000] [--limiar 0.5]
                                   [--teste 0.2] [--semente 42] [--grafico custo_logistica.png]
"""
import argparse
import numpy as np
import pandas as pd

ALVO = "infra_digital_basica"


# ----------------------------------------------------------------------------
# Núcleo
# ----------------------------------------------------------------------------
def sigmoide(z):
    """Sigmoide numericamente estável (evita overflow em exp para |z| grande)."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def custo(Xb, y, theta, eps=1e-12):
    p = np.clip(sigmoide(Xb @ theta), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def gradiente(Xb, y, theta):
    return Xb.T @ (sigmoide(Xb @ theta) - y) / len(y)


def treinar(Xb, y, alpha=1.0, n_iter=100000, tol=1e-13):
    theta = np.zeros(Xb.shape[1])
    hist = [custo(Xb, y, theta)]
    for it in range(1, n_iter + 1):
        theta = theta - alpha * gradiente(Xb, y, theta)
        hist.append(custo(Xb, y, theta))
        if abs(hist[-2] - hist[-1]) < tol:
            break
    return theta, np.array(hist), it


def prever_proba(Xb, theta):
    return sigmoide(Xb @ theta)


def prever(Xb, theta, limiar=0.5):
    return (prever_proba(Xb, theta) >= limiar).astype(int)


# ----------------------------------------------------------------------------
# Métricas (do zero)
# ----------------------------------------------------------------------------
def matriz_confusao(y, p):
    tn = int(np.sum((y == 0) & (p == 0))); fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0))); tp = int(np.sum((y == 1) & (p == 1)))
    return tn, fp, fn, tp


def metricas(y, p):
    tn, fp, fn, tp = matriz_confusao(y, p)
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(acuracia=acc, precisao=prec, recall=rec, f1=f1)


def auc_roc(y, proba):
    """AUC via estatística de Mann-Whitney (postos médios, trata empates)."""
    postos = pd.Series(proba).rank(method="average").to_numpy()
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((postos[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


# ----------------------------------------------------------------------------
def preparar(df):
    X = pd.DataFrame({
        "IN_LABORATORIO_INFORMATICA": df["IN_LABORATORIO_INFORMATICA"],
        "IN_BIBLIOTECA": df["IN_BIBLIOTECA"],
        "IN_ENERGIA_GERADOR_FOSSIL": df["IN_ENERGIA_GERADOR_FOSSIL"],
        "RURAL": (df["TP_LOCALIZACAO"] == 2).astype(int),
        "DEP_ESTADUAL": (df["TP_DEPENDENCIA"] == 2).astype(int),
        "DEP_FEDERAL": (df["TP_DEPENDENCIA"] == 1).astype(int),
        "DEP_PRIVADA": (df["TP_DEPENDENCIA"] == 4).astype(int),
    }).astype(float)
    return X, df[ALVO].to_numpy(int)


def divisao_estratificada(y, frac_teste, semente):
    rng = np.random.default_rng(semente)
    te = []
    for c in (0, 1):
        i = rng.permutation(np.where(y == c)[0])
        te.extend(i[: int(round(len(i) * frac_teste))])
    te = np.array(sorted(te))
    tr = np.setdiff1d(np.arange(len(y)), te)
    return tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--iter", type=int, default=100000)
    ap.add_argument("--limiar", type=float, default=0.5)
    ap.add_argument("--teste", type=float, default=0.2)
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--grafico")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    X, y = preparar(df)
    tr, te = divisao_estratificada(y, a.teste, a.semente)
    Xtr = np.hstack([np.ones((len(tr), 1)), X.iloc[tr].to_numpy()])
    Xte = np.hstack([np.ones((len(te), 1)), X.iloc[te].to_numpy()])

    theta, hist, iters = treinar(Xtr, y[tr], a.alpha, a.iter)

    print(f"Base: {len(df)} linhas | treino: {len(tr)} | teste: {len(te)} | "
          f"positivos: {y.mean():.1%} (estratificado)")
    print(f"alpha = {a.alpha} | iterações: {iters} | J: {hist[0]:.4f} -> {hist[-1]:.4f}\n")

    tab = pd.DataFrame({"parametro": ["intercepto"] + list(X.columns), "theta": theta,
                        "odds_ratio": np.exp(theta)})
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Conferência com scikit-learn (sem regularização), se disponível
    try:
        from sklearn.linear_model import LogisticRegression
        sk = LogisticRegression(C=1e12, max_iter=100000, tol=1e-12).fit(X.iloc[tr], y[tr])  # C enorme = sem regularização
        d = np.max(np.abs(np.concatenate([sk.intercept_, sk.coef_[0]]) - theta))
        print(f"\nConferência com scikit-learn: diferença máxima em θ = {d:.2e}")
    except ImportError:
        pass

    base = max(y[te].mean(), 1 - y[te].mean())
    for nome, Xs, ys in [("Treino", Xtr, y[tr]), ("Teste", Xte, y[te])]:
        p = prever(Xs, theta, a.limiar)
        m = metricas(ys, p)
        tn, fp, fn, tp = matriz_confusao(ys, p)
        print(f"\n{nome} (limiar {a.limiar}): acurácia = {m['acuracia']:.4f} | precisão = {m['precisao']:.4f} | "
              f"recall = {m['recall']:.4f} | F1 = {m['f1']:.4f} | AUC = {auc_roc(ys, prever_proba(Xs, theta)):.4f}")
        print(f"  matriz de confusão [[TN, FP], [FN, TP]] = [[{tn}, {fp}], [{fn}, {tp}]]")
    print(f"\nReferência: classificar tudo como classe majoritária = {base:.4f} de acurácia")

    # Efeito do limiar (escolhido no TREINO, avaliado no teste)
    limiares = np.linspace(0.05, 0.95, 91)
    f1s = [metricas(y[tr], prever(Xtr, theta, t))["f1"] for t in limiares]
    t_best = float(limiares[int(np.argmax(f1s))])
    m = metricas(y[te], prever(Xte, theta, t_best))
    print(f"\nLimiar que maximiza F1 no treino: {t_best:.2f} -> teste: acurácia = {m['acuracia']:.4f} | "
          f"precisão = {m['precisao']:.4f} | recall = {m['recall']:.4f} | F1 = {m['f1']:.4f}")

    if a.grafico:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
        # (a) início do treino: onde o custo realmente cai
        n0 = min(300, len(hist))
        ax.plot(range(n0), hist[:n0], lw=2)
        ax.set_title("Primeiras iterações")
        # (b) treino completo, eixo x em escala log (a cauda longa fica visível)
        ax2.plot(np.arange(1, len(hist)), hist[1:], lw=2)
        ax2.set_xscale("log")
        ax2.axhline(hist[-1], color="gray", ls="--", lw=1, label=f"J final = {hist[-1]:.4f}")
        ax2.set_title(f"Treino completo (eixo x log, α = {a.alpha})")
        ax2.legend()
        for x_ in (ax, ax2):
            x_.set_xlabel("iteração")
            x_.set_ylabel("custo J(θ) (entropia cruzada)")
            x_.grid(alpha=0.3)
        fig.suptitle("Regressão logística: convergência do gradiente descendente")
        fig.tight_layout()
        fig.savefig(a.grafico, dpi=150)
        print(f"\nGráfico salvo em {a.grafico}")


if __name__ == "__main__":
    main()
