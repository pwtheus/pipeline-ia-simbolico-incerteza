#!/usr/bin/env python3
"""
Compara as implementações do zero com o scikit-learn:

  A) Regressão linear multivariada (gradiente descendente) vs. LinearRegression  -> dados_etapa2.csv
  B) Ridge (L2) do zero vs. sklearn Ridge (mesmo λ = alpha do sklearn)             -> dados_etapa2.csv
  C) Regressão logística do zero vs. LogisticRegression                           -> dados_etapa1.csv
       C1) sem regularização (C = 1e12), comparação direta
       C2) configuração padrão do sklearn (L2 com C = 1), para mostrar a diferença

Deixe na mesma pasta: regressao_multivariada_gd.py, regressao_ridge.py, regressao_logistica.py.

Uso:
    python3 comparar_sklearn.py dados_etapa2.csv dados_etapa1.csv [--log]
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, roc_auc_score

import regressao_multivariada_gd as lin
import regressao_ridge as rid
import regressao_logistica as log

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
F = lambda v: f"{v:.6f}"


def cronometrar(fn):
    t = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t


# ----------------------------------------------------------------------------
def parte_linear(csv, usar_log, semente=42, teste=0.2):
    df = pd.read_csv(csv)
    nomes = [c for c in df.columns if c != lin.ALVO]
    X = df[nomes].to_numpy(float)
    y = df[lin.ALVO].to_numpy(float)
    if usar_log:
        y = np.log1p(y)
    idx = np.random.default_rng(semente).permutation(len(df))
    n_te = int(round(len(df) * teste))
    te, tr = idx[:n_te], idx[n_te:]
    alvo_txt = "log(1+y)" if usar_log else "y bruto"

    # --- A) Linear: GD do zero (variáveis padronizadas, θ convertido para escala original) ---
    def meu_gd():
        (Xs,), mu, sg = lin.padronizar(X[tr])
        th_s, _, it = lin.gradiente_descendente(lin.adicionar_bias(Xs), y[tr], alpha=0.1, n_iter=100000, tol=1e-15)
        return lin.desnormalizar(th_s, mu, sg), it
    (th_meu, it), t_meu = cronometrar(meu_gd)

    sk, t_sk = cronometrar(lambda: LinearRegression().fit(X[tr], y[tr]))
    th_sk = np.concatenate([[sk.intercept_], sk.coef_])

    pred_meu = lin.adicionar_bias(X[te]) @ th_meu
    pred_sk = sk.predict(X[te])

    print("=" * 78)
    print(f"A) REGRESSÃO LINEAR (alvo: {alvo_txt}) — {len(tr)} treino / {len(te)} teste")
    print("=" * 78)
    tab = pd.DataFrame({"parametro": ["intercepto"] + nomes, "GD do zero": th_meu, "sklearn": th_sk})
    tab["|dif|"] = (tab["GD do zero"] - tab["sklearn"]).abs()
    print(tab.to_string(index=False, float_format=F))
    print(f"\n{'':22s}{'GD do zero':>14s}{'sklearn':>14s}")
    print(f"{'MSE teste':22s}{mean_squared_error(y[te], pred_meu):>14.6f}{mean_squared_error(y[te], pred_sk):>14.6f}")
    print(f"{'R² teste':22s}{r2_score(y[te], pred_meu):>14.6f}{r2_score(y[te], pred_sk):>14.6f}")
    print(f"{'tempo (s)':22s}{t_meu:>14.4f}{t_sk:>14.4f}   (GD: {it} iterações)")
    print(f"Máx. diferença entre previsões: {np.max(np.abs(pred_meu - pred_sk)):.2e}")

    # --- B) Ridge do zero vs sklearn Ridge ---
    print("\n" + "-" * 78)
    print("B) RIDGE (L2) — mesmo λ nos dois (J = (1/2m)Σerr² + (λ/2m)Σθ²  ≡  sklearn alpha = λ)")
    print("-" * 78)
    (Xs_tr, Xs_te), mu, sg = lin.padronizar(X[tr], X[te])
    Xb_tr, Xb_te = lin.adicionar_bias(Xs_tr), lin.adicionar_bias(Xs_te)
    linhas = []
    for lam in [0.1, 10.0, 1000.0]:
        th_gd = rid.ridge_gd(Xb_tr, y[tr], lam, alpha=0.1, n_iter=100000, tol=1e-15)
        th_fc = rid.ridge_fechada(Xb_tr, y[tr], lam)
        skr = Ridge(alpha=lam).fit(Xs_tr, y[tr])  # nos dados padronizados, como acima
        th_sr = np.concatenate([[skr.intercept_], skr.coef_])
        linhas.append(dict(lam=lam,
                           **{"|GD - sklearn|": np.max(np.abs(th_gd - th_sr)),
                              "|fechada - sklearn|": np.max(np.abs(th_fc - th_sr)),
                              "MSE teste (meu)": mean_squared_error(y[te], Xb_te @ th_gd),
                              "MSE teste (sk)": mean_squared_error(y[te], skr.predict(Xs_te))}))
    print(pd.DataFrame(linhas).to_string(index=False, float_format=lambda v: f"{v:.3e}" if v < 1e-3 else f"{v:.6f}"))


# ----------------------------------------------------------------------------
def parte_logistica(csv, semente=42, teste=0.2):
    df = pd.read_csv(csv)
    X, y = log.preparar(df)
    tr, te = log.divisao_estratificada(y, teste, semente)
    Xtr_b = np.hstack([np.ones((len(tr), 1)), X.iloc[tr].to_numpy()])
    Xte_b = np.hstack([np.ones((len(te), 1)), X.iloc[te].to_numpy()])

    (th_meu, hist, it), t_meu = cronometrar(lambda: log.treinar(Xtr_b, y[tr], alpha=1.0, n_iter=300000, tol=1e-15))
    sk1, t1 = cronometrar(lambda: LogisticRegression(C=1e12, max_iter=100000, tol=1e-12).fit(X.iloc[tr], y[tr]))
    sk2, t2 = cronometrar(lambda: LogisticRegression().fit(X.iloc[tr], y[tr]))

    th1 = np.concatenate([sk1.intercept_, sk1.coef_[0]])
    th2 = np.concatenate([sk2.intercept_, sk2.coef_[0]])

    print("\n" + "=" * 78)
    print(f"C) REGRESSÃO LOGÍSTICA — {len(tr)} treino / {len(te)} teste (limiar 0,5)")
    print("=" * 78)
    tab = pd.DataFrame({"parametro": ["intercepto"] + list(X.columns), "do zero": th_meu,
                        "sk sem reg. (C=1e12)": th1, "sk padrão (L2, C=1)": th2})
    tab["|dif| vs sem reg."] = (tab["do zero"] - tab["sk sem reg. (C=1e12)"]).abs()
    print(tab.to_string(index=False, float_format=F))

    p_meu = log.prever_proba(Xte_b, th_meu)
    p1 = sk1.predict_proba(X.iloc[te])[:, 1]
    p2 = sk2.predict_proba(X.iloc[te])[:, 1]
    linhas = []
    for nome, p, t in [("do zero", p_meu, t_meu), ("sklearn sem reg.", p1, t1), ("sklearn padrão (L2)", p2, t2)]:
        c = (p >= 0.5).astype(int)
        linhas.append({"modelo": nome, "acurácia": accuracy_score(y[te], c), "F1": f1_score(y[te], c),
                       "AUC": roc_auc_score(y[te], p),
                       "custo teste": float(-np.mean(y[te] * np.log(np.clip(p, 1e-12, 1))
                                                     + (1 - y[te]) * np.log(np.clip(1 - p, 1e-12, 1)))),
                       "tempo (s)": t})
    print("\n" + pd.DataFrame(linhas).to_string(index=False, float_format=F))
    print(f"\nDo zero: {it} iterações.")
    print(f"Classes previstas iguais (do zero vs sklearn sem reg.): {np.mean((p_meu >= .5) == (p1 >= .5)):.2%}")
    print(f"Máx. diferença nas probabilidades (do zero vs sklearn sem reg.): {np.max(np.abs(p_meu - p1)):.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_etapa2")
    ap.add_argument("csv_etapa1")
    ap.add_argument("--log", action="store_true", help="usa log(1+y) como alvo na parte linear")
    a = ap.parse_args()
    parte_linear(a.csv_etapa2, a.log)
    parte_logistica(a.csv_etapa1)
