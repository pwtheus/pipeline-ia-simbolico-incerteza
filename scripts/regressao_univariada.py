#!/usr/bin/env python3
"""
Regressão linear univariada com fórmula fechada (mínimos quadrados ordinários).

Modelo:   ŷ = θ0 + θ1 · x

Fórmula fechada (derivada de ∂J/∂θ = 0, com J = (1/2m) Σ (ŷ - y)²):
    θ1 = Σ (xᵢ - x̄)(yᵢ - ȳ) / Σ (xᵢ - x̄)²   =  Cov(x, y) / Var(x)
    θ0 = ȳ - θ1 · x̄

Uso:
    python3 regressao_univariada.py base.csv [--feature IN_INTERNET_ALUNOS]
                                    [--log] [--teste 0.2] [--semente 42] [--grafico saida.png]

Sem --feature, usa internet para alunos, escolhida previamente pelo domínio.
Apenas NumPy/pandas na implementação (o scikit-learn só é usado, se existir, para conferir).
"""
import argparse
import numpy as np
import pandas as pd

ALVO = "DENSIDADE_INFORMATIZACAO"


# ----------------------------------------------------------------------------
# Implementação
# ----------------------------------------------------------------------------
def ajustar(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Retorna (θ0, θ1) pela fórmula fechada."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = x - x.mean()
    sxx = np.sum(dx ** 2)
    if sxx == 0:
        raise ValueError("x é constante: a variância é zero e θ1 não existe.")
    theta1 = np.sum(dx * (y - y.mean())) / sxx
    theta0 = y.mean() - theta1 * x.mean()
    return theta0, theta1


def prever(x: np.ndarray, theta0: float, theta1: float) -> np.ndarray:
    return theta0 + theta1 * np.asarray(x, dtype=float)


def mse(y, y_prev) -> float:
    return float(np.mean((np.asarray(y) - np.asarray(y_prev)) ** 2))


def r2(y, y_prev) -> float:
    y = np.asarray(y, dtype=float)
    ss_res = np.sum((y - y_prev) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def dividir(df: pd.DataFrame, frac_teste: float, semente: int):
    idx = np.random.default_rng(semente).permutation(len(df))
    n_teste = int(round(len(df) * frac_teste))
    return df.iloc[idx[n_teste:]], df.iloc[idx[:n_teste]]


# ----------------------------------------------------------------------------
# Execução
# ----------------------------------------------------------------------------
def avaliar(treino, teste, feature, usar_log):
    f = np.log1p if usar_log else (lambda v: v)
    xtr, ytr = treino[feature].to_numpy(float), f(treino[ALVO].to_numpy(float))
    xte, yte = teste[feature].to_numpy(float), f(teste[ALVO].to_numpy(float))
    t0, t1 = ajustar(xtr, ytr)
    ptr, pte = prever(xtr, t0, t1), prever(xte, t0, t1)
    corr = float(np.corrcoef(xtr, ytr)[0, 1])
    return dict(feature=feature, theta0=t0, theta1=t1, r_pearson=corr,
                mse_treino=mse(ytr, ptr), mse_teste=mse(yte, pte),
                rmse_teste=mse(yte, pte) ** 0.5, r2_treino=r2(ytr, ptr), r2_teste=r2(yte, pte))


def conferir_com_sklearn(treino, feature, usar_log, t0, t1):
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        return None
    y = treino[ALVO].to_numpy(float)
    y = np.log1p(y) if usar_log else y
    m = LinearRegression().fit(treino[[feature]].to_numpy(float), y)
    return abs(m.intercept_ - t0), abs(m.coef_[0] - t1)


def grafico(treino, feature, usar_log, t0, t1, caminho):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = treino[feature].to_numpy(float)
    y = treino[ALVO].to_numpy(float)
    y = np.log1p(y) if usar_log else y
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, y, s=8, alpha=0.25, label="escolas (treino)")
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, prever(xs, t0, t1), color="crimson", lw=2,
            label=f"ŷ = {t0:.3f} + {t1:.3f}·x")
    ax.set_xlabel(feature)
    ax.set_ylabel(("log(1 + " if usar_log else "") + "Densidade de Informatização" + (")" if usar_log else ""))
    ax.legend()
    ax.set_title("Regressão linear univariada (fórmula fechada)")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--feature", default="IN_INTERNET_ALUNOS")
    ap.add_argument("--log", action="store_true", help="usa log(1 + y) como alvo")
    ap.add_argument("--teste", type=float, default=0.2)
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--grafico")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    treino, teste = dividir(df, a.teste, a.semente)
    print(f"Base: {len(df)} linhas | treino: {len(treino)} | teste: {len(teste)} | "
          f"alvo: {'log(1+y)' if a.log else 'y (bruto)'}\n")

    feats = [a.feature]  # escolha anterior à avaliação; o teste não seleciona a pista
    res = pd.DataFrame([avaliar(treino, teste, f, a.log) for f in feats])
    print("Preditor definido antes da avaliação:", a.feature)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))

    melhor = res.iloc[0]
    print(f"\nModelo: ŷ = {melhor.theta0:.4f} + {melhor.theta1:.4f} · {melhor.feature}"
          f"   (R² teste = {melhor.r2_teste:.4f})")
    dif = conferir_com_sklearn(treino, melhor.feature, a.log, melhor.theta0, melhor.theta1)
    if dif:
        print(f"Conferência com scikit-learn: |Δθ0| = {dif[0]:.2e}, |Δθ1| = {dif[1]:.2e}")
    if a.grafico:
        grafico(treino, melhor.feature, a.log, melhor.theta0, melhor.theta1, a.grafico)
        print(f"Gráfico salvo em {a.grafico}")


if __name__ == "__main__":
    main()

