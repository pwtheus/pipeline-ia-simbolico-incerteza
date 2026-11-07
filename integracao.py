"""Demonstração operacional das saídas das Etapas 1, 4, 5 e 6, em teste separado."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skfuzzy.control import ControlSystemSimulation
from scripts.learn_decision_tree import treinar, prever
from scripts.naive_bayes import NaiveBayesDoZero
from fuzzy import construir_sistema, montar_rede, tabela_por_evidencia, ALVO


def executar_integracao(caminho='dados_etapa1.csv'):
    df = pd.read_csv(caminho)
    pistas = [c for c in df if c not in [ALVO, 'CO_ENTIDADE']]
    treino, teste = train_test_split(df, test_size=.3, random_state=42, stratify=df[ALVO])
    arvore = treinar(treino.to_dict('records'), pistas, ALVO, 3)
    nb = NaiveBayesDoZero()
    nb.fit(treino[pistas], treino[ALVO])
    rede = montar_rede(treino)
    sistema, variaveis = construir_sistema()
    tabela = tabela_por_evidencia(teste, treino, rede, pistas, sistema, variaveis)
    # Todas as cinco pistas são observadas neste cenário; não comparar com cadastro apenas.
    tabela['P_NB'] = nb.predict_proba(tabela[pistas])[:, list(nb.classes).index(1)]
    tabela['classe_arvore'] = [prever(arvore, r) for r in tabela.to_dict('records')]
    tabela['concordancia'] = 1 - abs(tabela.P_NB - tabela.P_infra)
    tabela['confianca_integrada'] = tabela.confianca * tabela.concordancia
    from fuzzy import calcular_urgencia, rotulo_urgencia
    sim = ControlSystemSimulation(sistema)
    tabela['urgencia_integrada'] = [calcular_urgencia(sim, r.risco, r.confianca_integrada) for r in tabela.itertuples()]
    tabela['rotulo_integrado'] = [rotulo_urgencia(u, variaveis[2]) for u in tabela.urgencia_integrada]
    tabela['revisar'] = tabela.classe_arvore != (tabela.P_infra >= .5).astype(int)
    tabela['acao_final'] = np.where(tabela.revisar, 'revisar divergência', tabela.rotulo_integrado)
    cols = pistas + ['P_infra','P_NB','classe_arvore','confianca_integrada','urgencia_integrada','acao_final']
    resultado = teste[['CO_ENTIDADE',ALVO]+pistas].merge(tabela[cols],on=pistas,validate='many_to_one')
    assert len(resultado) == len(teste)
    assert not set(treino.CO_ENTIDADE) & set(teste.CO_ENTIDADE)
    return resultado


if __name__ == '__main__':
    resultado = executar_integracao()
    print(resultado.head(10).to_string(index=False))
    print(resultado.acao_final.value_counts().to_string())
