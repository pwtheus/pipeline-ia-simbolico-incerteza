"""Mistura de dois perfis para atributos binários."""
import numpy as np


def responsabilidades(X, pesos, probabilidades):
    # Cada coluna representa um perfil. Cada linha representa uma escola.
    log_conjunta = (np.log(pesos) + X @ np.log(probabilidades).T
                    + (1 - X) @ np.log1p(-probabilidades).T)
    log_total = np.logaddexp(log_conjunta[:, 0], log_conjunta[:, 1])
    return np.exp(log_conjunta - log_total[:, None]), log_total.sum()


def ajustar_em(X, max_iter=500, tolerancia=1e-7):
    pesos = np.array([0.5, 0.5])
    probabilidades = np.random.default_rng(42).uniform(0.2, 0.8, (2, X.shape[1]))
    historico = []
    for _ in range(max_iter):
        r, log_verossimilhanca = responsabilidades(X, pesos, probabilidades)
        historico.append(log_verossimilhanca)
        if len(historico) > 1 and abs(historico[-1] - historico[-2]) / len(X) < tolerancia:
            break
        if len(historico) < max_iter:
            totais = r.sum(axis=0)
            pesos = totais / len(X)
            probabilidades = np.clip((r.T @ X) / totais[:, None], 1e-9, 1 - 1e-9)
    return pesos, probabilidades, historico


def prever_perfil(X, pesos, probabilidades):
    r, _ = responsabilidades(X, pesos, probabilidades)
    return r.argmax(axis=1)
