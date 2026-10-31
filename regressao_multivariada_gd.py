#!/usr/bin/env python3
"""
Regressão linear multivariada com gradiente descendente (implementada do zero, só NumPy).

Modelo:      ŷ = X θ,   com X = [1, x1, ..., xn]  (coluna de 1s para o intercepto θ0)
Custo:       J(θ) = (1 / 2m) · Σ (ŷᵢ - yᵢ)²
Gradiente:   ∇J(θ) = (1 / m) · Xᵀ (X θ - y)
Atualização: θ ← θ - α · ∇J(θ)

As preditoras são padronizadas (z-score, com média/desvio do TREINO) para que o
gradiente descendente convirja bem. Ao final, θ é convertido de volta para a escala original.

Uso:
    python3 regressao_multivariada_gd.py dados_etapa2.csv [--log] [--alpha 0.1] [--iter 2000]
                                         [--teste 0.2] [--semente 42] [--grafico custo.png]
"""
import argparse
import numpy as np
import pandas as pd

ALVO = "DENSIDADE_INFORMATIZACAO"


# ----------------------------------------------------------------------------
# Núcleo: funções do zero
# ----------------------------------------------------------------------------
def adicionar_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1)), X])


def custo(Xb: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    erro = Xb @ theta - y
    return float(erro @ erro / (2 * len(y)))


def gradiente(Xb: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return Xb.T @ (Xb @ theta - y) / len(y)


def gradiente_descendente(Xb, y, alpha=0.1, n_iter=2000, tol=1e-10):
    """Retorna (θ, histórico_do_custo, iterações_realizadas). Para se ΔJ < tol."""
    theta = np.zeros(Xb.shape[1])
    historico = [custo(Xb, y, theta)]
    for it in range(1, n_iter + 1):
        theta = theta - alpha * gradiente(Xb, y, theta)
        historico.append(custo(Xb, y, theta))
        if not np.isfinite(historico[-1]):
            raise FloatingPointError(f"Divergiu com alpha={alpha}. Reduza a taxa de aprendizado.")
        if abs(historico[-2] - historico[-1]) < tol:
            break
    return theta, np.array(historico), it


def padronizar(X_treino, *outros):
    mu = X_treino.mean(axis=0)
    sigma = X_treino.std(axis=0)
    sigma[sigma == 0] = 1.0
    return [(A - mu) / sigma for A in (X_treino, *outros)], mu, sigma


def desnormalizar(theta, mu, sigma):
    """Converte θ (treinado nas variáveis padronizadas) para a escala original."""
    coef = theta[1:] / sigma
    intercepto = theta[0] - np.sum(theta[1:] * mu / sigma)
    return np.concatenate([[intercepto], coef])


def mse(y, p): return float(np.mean((y - p) ** 2))
def r2(y, p): return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def equacao_normal(Xb, y):
    """Solução fechada, usada só para conferir o resultado do gradiente descendente."""
    return np.linalg.lstsq(Xb, y, rcond=None)[0]


# ----------------------------------------------------------------------------
# Execução
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--log", action="store_true", help="usa log(1 + y) como alvo")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--iter", type=int, default=2000)
    ap.add_argument("--teste", type=float, default=0.2)
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--grafico")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    preditoras = [c for c in df.columns if c != ALVO]
    X = df[preditoras].to_numpy(float)
    y = df[ALVO].to_numpy(float)
    if a.log:
        y = np.log1p(y)

    idx = np.random.default_rng(a.semente).permutation(len(df))
    n_te = int(round(len(df) * a.teste))
    te, tr = idx[:n_te], idx[n_te:]

    (Xtr_s, Xte_s), mu, sigma = padronizar(X[tr], X[te])
    Xtr_b, Xte_b = adicionar_bias(Xtr_s), adicionar_bias(Xte_s)

    theta_s, hist, iters = gradiente_descendente(Xtr_b, y[tr], a.alpha, a.iter)
    theta = desnormalizar(theta_s, mu, sigma)

    p_tr, p_te = Xtr_b @ theta_s, Xte_b @ theta_s

    # Conferência com a equação normal
    theta_ref = desnormalizar(equacao_normal(Xtr_b, y[tr]), mu, sigma)

    print(f"Base: {len(df)} linhas | treino: {len(tr)} | teste: {len(te)} | "
          f"alvo: {'log(1+y)' if a.log else 'y (bruto)'}")
    print(f"alpha = {a.alpha} | iterações realizadas: {iters} | "
          f"J inicial = {hist[0]:.4f} -> J final = {hist[-1]:.4f}\n")

    nomes = ["intercepto"] + preditoras
    tabela = pd.DataFrame({"parametro": nomes, "theta_GD": theta, "theta_eq_normal": theta_ref})
    tabela["|diferenca|"] = (tabela.theta_GD - tabela.theta_eq_normal).abs()
    pd.set_option("display.width", 200)
    print(tabela.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    print(f"\nTreino: MSE = {mse(y[tr], p_tr):.4f} | R² = {r2(y[tr], p_tr):.4f}")
    print(f"Teste : MSE = {mse(y[te], p_te):.4f} | RMSE = {mse(y[te], p_te) ** 0.5:.4f} | "
          f"R² = {r2(y[te], p_te):.4f}")

    if a.grafico:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(hist, lw=2)
        ax.set_xlabel("iteração")
        ax.set_ylabel("custo J(θ)")
        ax.set_title(f"Convergência do gradiente descendente (α = {a.alpha})")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(a.grafico, dpi=150)
        print(f"\nGráfico salvo em {a.grafico}")


if __name__ == "__main__":
    main()
