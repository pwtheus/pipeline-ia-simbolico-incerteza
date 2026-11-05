"""Estimação das CPTs da rede Bayesiana por contagem + suavização de Laplace.

Uso:
    python estimar_cpts.py dados_etapa1.csv [pasta_saida] [alpha]

Sem dependências além de pandas.
"""
import itertools
import sys
from pathlib import Path

import pandas as pd

# Estrutura do DAG: nó -> lista de pais (mesma ordem usada nas tabelas)
PAIS = {
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
LIMIAR_POUCAS_OBS = 30  # configurações de pais com menos linhas são sinalizadas


def estimar_cpt(df, no, pais, estados, alpha):
    """Retorna (CPT, contagens_dos_pais). Linhas = estados do nó; colunas = configs dos pais."""
    estados_no = estados[no]
    if not pais:
        cont = df[no].value_counts().reindex(estados_no, fill_value=0)
        n = cont.sum()
        cpt = (cont + alpha) / (n + alpha * len(estados_no))
        return cpt.to_frame("P"), pd.Series({"(sem pais)": n})

    configs = list(itertools.product(*[estados[p] for p in pais]))
    cont = df.groupby(pais + [no]).size()
    cpt = pd.DataFrame(index=estados_no, columns=pd.MultiIndex.from_tuples(configs, names=pais), dtype=float)
    n_pais = {}
    for cfg in configs:
        n_cfg = 0
        contagens = []
        for e in estados_no:
            c = int(cont.get(cfg + (e,), 0))
            contagens.append(c)
            n_cfg += c
        for e, c in zip(estados_no, contagens):
            cpt.loc[e, cfg] = (c + alpha) / (n_cfg + alpha * len(estados_no))
        n_pais[cfg] = n_cfg
    return cpt, pd.Series(n_pais, name="N")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    caminho = Path(sys.argv[1])
    saida = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("cpts")
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    saida.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(caminho)
    colunas = list(PAIS)
    faltam = [c for c in colunas if c not in df.columns]
    if faltam:
        sys.exit(f"Colunas ausentes no CSV: {faltam}")
    antes = len(df)
    df = df[colunas].dropna().copy()
    print(f"Linhas: {antes} | após remover faltantes: {len(df)} | alpha = {alpha}\n")

    estados = {c: sorted(df[c].unique().tolist()) for c in colunas}
    for c in colunas:
        print(f"{c}: estados = {estados[c]}")
    print()

    for no, pais in PAIS.items():
        cpt, n_pais = estimar_cpt(df, no, pais, estados, alpha)
        soma = cpt.sum(axis=0)
        assert ((soma - 1).abs() < 1e-9).all(), f"CPT de {no} não soma 1"
        print("=" * 70)
        print(f"P({no} | {', '.join(pais) if pais else '—'})")
        print(cpt.round(4).to_string())
        print("\nN por configuração dos pais:")
        print(n_pais.to_string())
        raras = n_pais[n_pais < LIMIAR_POUCAS_OBS]
        if len(raras) and pais:
            print(f"ATENÇÃO: {len(raras)} configuração(ões) com menos de {LIMIAR_POUCAS_OBS} observações: {list(raras.index)}")
        print()
        cpt.round(6).to_csv(saida / f"cpt_{no}.csv")
        n_pais.to_csv(saida / f"n_pais_{no}.csv", header=True)
    print(f"Arquivos salvos em: {saida.resolve()}")


if __name__ == "__main__":
    main()
