#!/usr/bin/env python3
"""
Regressão Ridge (regularização L2) do zero + curva de coeficientes vs. λ.

Custo:       J(θ) = (1/2m) Σ (ŷᵢ - yᵢ)²  +  (λ/2m) Σ_{j≥1} θⱼ²      (o intercepto θ0 NÃO é penalizado)
Gradiente:   ∇J = (1/m) Xᵀ(Xθ - y) + (λ/m) [0, θ1, ..., θn]
Fechada:     θ = (XᵀX + λ·I₀)⁻¹ Xᵀy      (I₀ = identidade com o 1º elemento zerado)

Reaproveita as funções do gradiente descendente de regressao_multivariada_gd.py
(deixe os dois arquivos na mesma pasta).

Uso:
    python3 regressao_ridge.py dados_etapa2.csv [--log] [--saida-grafico ridge_coeficientes.png]
"""
import argparse
import numpy as np
import pandas as pd

from regressao_multivariada_gd import (ALVO, adicionar_bias, custo, padronizar, mse, r2)


# ----------------------------------------------------------------------------
# Ridge
# ----------------------------------------------------------------------------
def custo_ridge(Xb, y, theta, lam):
    return custo(Xb, y, theta) + lam * np.sum(theta[1:] ** 2) / (2 * len(y))


def gradiente_ridge(Xb, y, theta, lam):
    reg = lam * theta / len(y)
    reg[0] = 0.0  # não penaliza o intercepto
    return Xb.T @ (Xb @ theta - y) / len(y) + reg


def ridge_gd(Xb, y, lam, alpha=0.1, n_iter=20000, tol=1e-14):
    theta = np.zeros(Xb.shape[1])
    anterior = custo_ridge(Xb, y, theta, lam)
    for _ in range(n_iter):
        theta = theta - alpha * gradiente_ridge(Xb, y, theta, lam)
        atual = custo_ridge(Xb, y, theta, lam)
        if not np.isfinite(atual):
            raise FloatingPointError(f"Divergiu (alpha={alpha}, lambda={lam}).")
        if abs(anterior - atual) < tol:
            break
        anterior = atual
    return theta


def ridge_fechada(Xb, y, lam):
    I0 = np.eye(Xb.shape[1])
    I0[0, 0] = 0.0
    return np.linalg.solve(Xb.T @ Xb + lam * I0, Xb.T @ y)


def cv_mse(X, y, lam, k=5, semente=0):
    """MSE médio de validação cruzada k-fold (padroniza dentro de cada dobra)."""
    idx = np.random.default_rng(semente).permutation(len(y))
    erros = []
    for f in range(k):
        val = idx[f::k]
        trn = np.setdiff1d(idx, val)
        (Xt, Xv), _, _ = padronizar(X[trn], X[val])
        th = ridge_fechada(adicionar_bias(Xt), y[trn], lam)
        erros.append(mse(y[val], adicionar_bias(Xv) @ th))
    return float(np.mean(erros))


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--log", action="store_true", help="usa log(1 + y) como alvo")
    ap.add_argument("--teste", type=float, default=0.2)
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--saida-grafico", default="ridge_coeficientes.png")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    nomes = [c for c in df.columns if c != ALVO]
    X = df[nomes].to_numpy(float)
    y = df[ALVO].to_numpy(float)
    if a.log:
        y = np.log1p(y)

    idx = np.random.default_rng(a.semente).permutation(len(df))
    n_te = int(round(len(df) * a.teste))
    te, tr = idx[:n_te], idx[n_te:]
    (Xtr_s, Xte_s), _, _ = padronizar(X[tr], X[te])
    Xtr_b, Xte_b = adicionar_bias(Xtr_s), adicionar_bias(Xte_s)

    # 1) Validação: gradiente descendente vs. solução fechada
    print("Validação (GD com L2 vs. solução fechada), diferença máxima entre θ:")
    for lam in [0.0, 10.0, 1000.0]:
        d = np.max(np.abs(ridge_gd(Xtr_b, y[tr], lam) - ridge_fechada(Xtr_b, y[tr], lam)))
        print(f"  λ = {lam:>7g}: {d:.2e}")

    # 2) Caminho dos coeficientes
    lambdas = np.logspace(-2, 5, 100)
    caminho = np.array([ridge_fechada(Xtr_b, y[tr], l)[1:] for l in lambdas])
    theta_ols = ridge_fechada(Xtr_b, y[tr], 0.0)[1:]

    # 3) Escolha de λ por validação cruzada (apenas no treino)
    cv = np.array([cv_mse(X[tr], y[tr], l) for l in lambdas])
    i_best = int(np.argmin(cv))
    lam_best = lambdas[i_best]

    # 4) Resultados no teste
    def teste(lam):
        th = ridge_fechada(Xtr_b, y[tr], lam)
        return mse(y[te], Xte_b @ th), r2(y[te], Xte_b @ th)

    print(f"\nλ ótimo por CV (5 dobras): {lam_best:.4g}  (MSE CV = {cv[i_best]:.5f})")
    print(f"\n{'λ':>10s} {'MSE teste':>12s} {'R² teste':>10s}")
    for lam in [0.0, lam_best, 100, 1000, 10000]:
        m_, r_ = teste(lam)
        marca = "  <- ótimo CV" if lam == lam_best else ""
        print(f"{lam:>10.4g} {m_:>12.5f} {r_:>10.4f}{marca}")

    tab = pd.DataFrame({"preditora": nomes, "theta_OLS (λ=0)": theta_ols,
                        f"theta_Ridge (λ={lam_best:.3g})": ridge_fechada(Xtr_b, y[tr], lam_best)[1:]})
    print("\nCoeficientes (variáveis padronizadas):")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    # 5) Gráficos
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for j, n in enumerate(nomes):
        ax1.plot(lambdas, caminho[:, j], lw=2, label=n)
    ax1.set_xscale("log")
    ax1.axhline(0, color="gray", lw=0.8)
    ax1.axvline(lam_best, color="black", ls="--", lw=1, label=f"λ ótimo CV = {lam_best:.3g}")
    ax1.set_xlabel("λ (escala log)")
    ax1.set_ylabel("coeficiente (variáveis padronizadas)")
    ax1.set_title("Ridge: coeficientes vs. λ")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    ax2.plot(lambdas, cv, lw=2, color="tab:purple")
    ax2.axvline(lam_best, color="black", ls="--", lw=1)
    ax2.set_xscale("log")
    ax2.set_xlabel("λ (escala log)")
    ax2.set_ylabel("MSE de validação cruzada")
    ax2.set_title("Escolha de λ")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(a.saida_grafico, dpi=150)
    print(f"\nGráfico salvo em {a.saida_grafico}")


if __name__ == "__main__":
    main()
