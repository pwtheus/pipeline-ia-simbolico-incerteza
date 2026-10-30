>>> Work in progress


> Claude implementa a LEARN-DECISION-TREE


Implementei o LEARN-DECISION-TREE do zero no arquivo `learn\_decision\_tree.py`, usando só a biblioteca padrão do Python (sem pandas nem sklearn).

**Como funciona (pseudocódigo do AIMA/ID3)**

A função recursiva `learn\_decision\_tree(exemplos, atributos, exemplos\_pai)` segue quatro casos, nesta ordem:

1. **Sem exemplos:** devolve a classe mais frequente do pai (`valor\_plural`). 

2. **Todos da mesma classe:** devolve uma folha pura. 

3. **Sem atributos restantes (ou profundidade máxima):** devolve a classe mais frequente do nó. 

4. **Caso geral:** escolhe o atributo de maior ganho de informação e cria um ramo para cada valor dele (divisão multiway). Cada ramo chama a função de novo, sem esse atributo. 

As três peças de base são as mesmas fórmulas do seu documento: `entropia` (H(S) com log₂), `ganho\_informacao` (GI(S, A)) e `valor\_plural`. Para o caso 1 aparecer de verdade, o código guarda o domínio completo de cada atributo e cria um ramo para cada valor, mesmo que ele não ocorra no nó.

**Validação**

Não tenho o `dados\_etapa1.csv`, então testei com dados sintéticos que reproduzem as contagens da raiz. O resultado bateu com o seu cálculo manual: H(S) = 0,6824 e GI(IN\_LABORATORIO\_INFORMATICA) = 0,1915, escolhida como raiz. Como os outros atributos eram aleatórios, os níveis seguintes do teste não valem como comparação.

**Como rodar com os seus dados**

```
`python learn\_decision\_tree.py dados\_etapa1.csv --max-depth 3`
```

Com `--max-depth 3` a árvore para nos 3 níveis do seu esboço e imprime os ganhos da raiz, a árvore indentada (com n e contagem de cada nó) e a acurácia no treino. Sem esse parâmetro ela cresce até as folhas puras. Com os dados reais, vale conferir se os níveis 2 e 3 reproduzem o seu documento: TP\_LOCALIZACAO em Lab=0, TP\_DEPENDENCIA em Lab=1 e GI 0,0347 em Lab=1 com Dependência=3.

Dois pontos para o relatório: a acurácia no treino de uma árvore sem limite de profundidade é otimista (ela tende a decorar), então vale separar treino e teste. E, como você anotou, o ganho de informação favorece atributos com muitos valores. Se o trabalho pedir, dá para trocar `ganho\_informacao` por razão de ganho mexendo só nessa função.


> Treinando a árvore com os dados gerados:

No terminal, entrar com:
python learn_decision_tree.py dados_etapa1.csv --max-depth 3

Saída:
H(raiz) = 0.6824
  GI(IN_LABORATORIO_INFORMATICA) = 0.1915
  GI(IN_BIBLIOTECA) = 0.0989
  GI(TP_LOCALIZACAO) = 0.0953
  GI(TP_DEPENDENCIA) = 0.0438
  GI(IN_ENERGIA_GERADOR_FOSSIL) = 0.0088

[IN_LABORATORIO_INFORMATICA]  n=5562, {'0': 4555, '1': 1007}
  IN_LABORATORIO_INFORMATICA = 0.0:
    [TP_LOCALIZACAO]  n=4607, {'0': 4238, '1': 369}
      TP_LOCALIZACAO = 1:
        [TP_DEPENDENCIA]  n=1107, {'0': 926, '1': 181}
          TP_DEPENDENCIA = 1:
            -> classe 0  (n=2, {'1': 1, '0': 1})
          TP_DEPENDENCIA = 2:
            -> classe 0  (n=220, {'0': 179, '1': 41})
          TP_DEPENDENCIA = 3:
            -> classe 0  (n=608, {'0': 532, '1': 76})
          TP_DEPENDENCIA = 4:
            -> classe 0  (n=277, {'1': 63, '0': 214})
      TP_LOCALIZACAO = 2:
        [TP_DEPENDENCIA]  n=3500, {'0': 3312, '1': 188}
          TP_DEPENDENCIA = 1:
            -> classe 0  (n=0, {})
          TP_DEPENDENCIA = 2:
            -> classe 0  (n=137, {'0': 135, '1': 2})
          TP_DEPENDENCIA = 3:
            -> classe 0  (n=3360, {'0': 3175, '1': 185})
          TP_DEPENDENCIA = 4:
            -> classe 0  (n=3, {'0': 2, '1': 1})
  IN_LABORATORIO_INFORMATICA = 1.0:
    [TP_DEPENDENCIA]  n=955, {'0': 317, '1': 638}
      TP_DEPENDENCIA = 1:
        -> classe 1  (n=17, {'1': 17})
      TP_DEPENDENCIA = 2:
        [IN_ENERGIA_GERADOR_FOSSIL]  n=401, {'0': 189, '1': 212}
          IN_ENERGIA_GERADOR_FOSSIL = 0.0:
            -> classe 1  (n=400, {'0': 188, '1': 212})
          IN_ENERGIA_GERADOR_FOSSIL = 1.0:
            -> classe 0  (n=1, {'0': 1})
      TP_DEPENDENCIA = 3:
        [TP_LOCALIZACAO]  n=413, {'0': 99, '1': 314}
          TP_LOCALIZACAO = 1:
            -> classe 1  (n=285, {'0': 50, '1': 235})
          TP_LOCALIZACAO = 2:
            -> classe 1  (n=128, {'0': 49, '1': 79})
      TP_DEPENDENCIA = 4:
        [IN_ENERGIA_GERADOR_FOSSIL]  n=124, {'1': 95, '0': 29}
          IN_ENERGIA_GERADOR_FOSSIL = 0.0:
            -> classe 1  (n=116, {'1': 87, '0': 29})
          IN_ENERGIA_GERADOR_FOSSIL = 1.0:
            -> classe 1  (n=8, {'1': 8})

Acurácia no treino: 0.8768


> Claude compara a árvore gerada com DecisionTreeClassifier do scikit-learn, usando os dados_etapa1.csv

Rodei a comparação com o dados_etapa1.csv (5562 escolas): o ID3 próprio e o DecisionTreeClassifier tiveram a mesma acurácia e as mesmas previsões, e só a estrutura das árvores difere.

Conferência com o seu cálculo manual (dados completos, max_depth=3)

Meu código reproduziu todos os valores do seu documento: H(S) = 0,6824 e GI(Lab) = 0,1915 na raiz. No nível 2, TP_LOCALIZACAO (0,0187) em Lab=0 e TP_DEPENDENCIA (0,0525) em Lab=1. No nível 3, 0,0347 em Lab=1 e Dependência=3, e 0,0258 em Dependência=4, entre outros.

Comparação com treino/teste (70/30 estratificado, seed 42)

max_depth	Modelo	Profundidade	Folhas	Acc. treino	Acc. teste
3	ID3 próprio (multiway)	3	13	0,8728	0,8868
3	sklearn (entropy)	3	8	0,8728	0,8868
3	sklearn (gini)	3	8	0,8728	0,8868
sem limite	ID3 próprio	5	47	0,8739	0,8838
sem limite	sklearn (entropy)	7	33	0,8739	0,8838
sem limite	sklearn (gini)	7	33	0,8739	0,8838

A concordância das previsões no teste foi de 100% entre os três modelos nos dois cenários. Com todos os dados, a acurácia de treino também foi igual nos dois (0,8768), com 100% de concordância. Como referência, prever sempre a classe 0 dá cerca de 0,82.

O que observar na análise

Mesma acurácia, estruturas diferentes: o sklearn só faz divisões binárias. Em Lab=1, ele separa Dependência=2 contra as demais (1, 3 e 4 juntas), enquanto o seu ID3 abre os quatro valores de uma vez. Por isso, com max_depth=3, o ID3 tem 13 folhas e o sklearn 8.
Escolha de atributo diferente no nível 3: em Lab=0 e Localização=1, o seu ID3 escolhe TP_DEPENDENCIA (GI 0,0108), e o sklearn escolhe IN_BIBLIOTECA. Com a divisão binária (one-hot), o ganho de Dependência cai abaixo do de Biblioteca (GI 0,0101 no seu cálculo multiway). É um bom exemplo de como a forma da divisão muda o critério.
Entropia × Gini: produziram a mesma árvore neste problema. Para este dataset, a escolha do critério não importou.
Sem sobreajuste visível: ir de profundidade 3 para árvore completa quase não muda o treino (0,8728 → 0,8739) e não melhora o teste (0,8868 → 0,8838). Há só 5 atributos e muitas escolas com a mesma combinação, então a árvore não tem como aprofundar de forma útil.
Ramos vazios: na partição de treino apareceu o caso 1 do algoritmo. Em Lab=0 e Localização=2, TP_DEPENDENCIA=1 ficou com n = 0 e recebeu a classe majoritária do pai.
Tempo: o ID3 em Python puro levou cerca de 0,04 s e o sklearn cerca de 0,005 s. É irrelevante nesta base, mas dá para citar.


> Rodando a comparação:

No terminal:
python comparar_sklearn.py dados_etapa1.csv

Saída:
Treino: 3893 | Teste: 1669

========================================================================
max_depth=3
========================================================================
modelo                       raiz                           prof folhas  acc_tr  acc_te  tempo(s)
ID3 próprio (multiway)       IN_LABORATORIO_INFORMATICA        3     13  0.8728  0.8868    0.0352
sklearn (entropy, one-hot)   IN_LABORATORIO_INFORMATICA=1.0    3      8  0.8728  0.8868    0.0045
sklearn (gini, one-hot)      IN_LABORATORIO_INFORMATICA=1.0    3      8  0.8728  0.8868    0.0028

Concordância das previsões no teste:
  próprio x sklearn-entropy: 1.0000
  próprio x sklearn-gini   : 1.0000
  sklearn-entropy x gini   : 1.0000

Árvore do sklearn (entropy, max_depth=3):
|--- IN_LABORATORIO_INFORMATICA=1.0 <= 0.50
|   |--- TP_LOCALIZACAO=1 <= 0.50
|   |   |--- TP_DEPENDENCIA=2 <= 0.50
|   |   |   |--- class: 0
|   |   |--- TP_DEPENDENCIA=2 >  0.50
|   |   |   |--- class: 0
|   |--- TP_LOCALIZACAO=1 >  0.50
|   |   |--- IN_BIBLIOTECA=1.0 <= 0.50
|   |   |   |--- class: 0
|   |   |--- IN_BIBLIOTECA=1.0 >  0.50
|   |   |   |--- class: 0
|--- IN_LABORATORIO_INFORMATICA=1.0 >  0.50
|   |--- TP_DEPENDENCIA=2 <= 0.50
|   |   |--- TP_LOCALIZACAO=2 <= 0.50
|   |   |   |--- class: 1
|   |   |--- TP_LOCALIZACAO=2 >  0.50
|   |   |   |--- class: 1
|   |--- TP_DEPENDENCIA=2 >  0.50
|   |   |--- TP_LOCALIZACAO=2 <= 0.50
|   |   |   |--- class: 1
|   |   |--- TP_LOCALIZACAO=2 >  0.50
|   |   |   |--- class: 0

Árvore própria (max_depth=3):
[IN_LABORATORIO_INFORMATICA]  n=3893, {'0': 3188, '1': 705}
  IN_LABORATORIO_INFORMATICA = 0.0:
    [TP_LOCALIZACAO]  n=3234, {'0': 2962, '1': 272}
      TP_LOCALIZACAO = 1:
        [IN_BIBLIOTECA]  n=801, {'0': 668, '1': 133}
          IN_BIBLIOTECA = 0.0:
            -> classe 0  (n=490, {'0': 431, '1': 59})
          IN_BIBLIOTECA = 1.0:
            -> classe 0  (n=311, {'0': 237, '1': 74})
      TP_LOCALIZACAO = 2:
        [TP_DEPENDENCIA]  n=2433, {'0': 2294, '1': 139}
          TP_DEPENDENCIA = 1:
            -> classe 0  (n=0, {})
          TP_DEPENDENCIA = 2:
            -> classe 0  (n=102, {'0': 101, '1': 1})
          TP_DEPENDENCIA = 3:
            -> classe 0  (n=2329, {'0': 2192, '1': 137})
          TP_DEPENDENCIA = 4:
            -> classe 0  (n=2, {'1': 1, '0': 1})
  IN_LABORATORIO_INFORMATICA = 1.0:
    [TP_DEPENDENCIA]  n=659, {'0': 226, '1': 433}
      TP_DEPENDENCIA = 1:
        -> classe 1  (n=9, {'1': 9})
      TP_DEPENDENCIA = 2:
        [TP_LOCALIZACAO]  n=280, {'0': 137, '1': 143}
          TP_LOCALIZACAO = 1:
            -> classe 1  (n=261, {'0': 126, '1': 135})
          TP_LOCALIZACAO = 2:
            -> classe 0  (n=19, {'0': 11, '1': 8})
      TP_DEPENDENCIA = 3:
        [TP_LOCALIZACAO]  n=283, {'0': 70, '1': 213}
          TP_LOCALIZACAO = 1:
            -> classe 1  (n=195, {'0': 37, '1': 158})
          TP_LOCALIZACAO = 2:
            -> classe 1  (n=88, {'0': 33, '1': 55})
      TP_DEPENDENCIA = 4:
        [IN_BIBLIOTECA]  n=87, {'1': 68, '0': 19}
          IN_BIBLIOTECA = 0.0:
            -> classe 1  (n=14, {'0': 6, '1': 8})
          IN_BIBLIOTECA = 1.0:
            -> classe 1  (n=73, {'1': 60, '0': 13})

========================================================================
max_depth=None
========================================================================
modelo                       raiz                           prof folhas  acc_tr  acc_te  tempo(s)
ID3 próprio (multiway)       IN_LABORATORIO_INFORMATICA        5     47  0.8739  0.8838    0.0672
sklearn (entropy, one-hot)   IN_LABORATORIO_INFORMATICA=1.0    7     33  0.8739  0.8838    0.0026
sklearn (gini, one-hot)      IN_LABORATORIO_INFORMATICA=1.0    7     33  0.8739  0.8838    0.0023

Concordância das previsões no teste:
  próprio x sklearn-entropy: 1.0000
  próprio x sklearn-gini   : 1.0000
  sklearn-entropy x gini   : 1.0000

========================================================================
Dados completos (n = 5562), max_depth=3: confere com o cálculo manual
========================================================================
H(raiz) = 0.6824
  nivel 1 n=5562 {'0': 4555, '1': 1007} -> IN_LABORATORIO_INFORMATICA (GI=0.1915) | IN_BIBLIOTECA=0.0989; TP_LOCALIZACAO=0.0953; TP_DEPENDENCIA=0.0438; IN_ENERGIA_GERADOR_FOSSIL=0.0088
  nivel 2 n=4607 {'0': 4238, '1': 369} -> TP_LOCALIZACAO (GI=0.0187) | IN_BIBLIOTECA=0.0130; TP_DEPENDENCIA=0.0124; IN_ENERGIA_GERADOR_FOSSIL=0.0018
  nivel 3 n=1107 {'0': 926, '1': 181} -> TP_DEPENDENCIA (GI=0.0108) | IN_BIBLIOTECA=0.0101; IN_ENERGIA_GERADOR_FOSSIL=0.0047
  nivel 3 n=3500 {'0': 3312, '1': 188} -> TP_DEPENDENCIA (GI=0.0017) | IN_BIBLIOTECA=0.0016; IN_ENERGIA_GERADOR_FOSSIL=0.0004
  nivel 2 n= 955 {'0': 317, '1': 638} -> TP_DEPENDENCIA (GI=0.0525) | TP_LOCALIZACAO=0.0040; IN_ENERGIA_GERADOR_FOSSIL=0.0019; IN_BIBLIOTECA=0.0000
  nivel 3 n= 401 {'0': 189, '1': 212} -> IN_ENERGIA_GERADOR_FOSSIL (GI=0.0027) | IN_BIBLIOTECA=0.0019; TP_LOCALIZACAO=0.0009
  nivel 3 n= 413 {'0': 99, '1': 314} -> TP_LOCALIZACAO (GI=0.0347) | IN_BIBLIOTECA=0.0094; IN_ENERGIA_GERADOR_FOSSIL=0.0006
  nivel 3 n= 124 {'1': 95, '0': 29} -> IN_ENERGIA_GERADOR_FOSSIL (GI=0.0258) | TP_LOCALIZACAO=0.0171; IN_BIBLIOTECA=0.0163

acc treino  próprio=0.8768  sklearn-entropy=0.8768
concordância próprio x sklearn: 1.0000
|--- IN_LABORATORIO_INFORMATICA=1.0 <= 0.50
|   |--- TP_LOCALIZACAO=1 <= 0.50
|   |   |--- IN_BIBLIOTECA=0.0 <= 0.50
|   |   |   |--- class: 0
|   |   |--- IN_BIBLIOTECA=0.0 >  0.50
|   |   |   |--- class: 0
|   |--- TP_LOCALIZACAO=1 >  0.50
|   |   |--- IN_BIBLIOTECA=1.0 <= 0.50
|   |   |   |--- class: 0
|   |   |--- IN_BIBLIOTECA=1.0 >  0.50
|   |   |   |--- class: 0
|--- IN_LABORATORIO_INFORMATICA=1.0 >  0.50
|   |--- TP_DEPENDENCIA=2 <= 0.50
|   |   |--- TP_LOCALIZACAO=2 <= 0.50
|   |   |   |--- class: 1
|   |   |--- TP_LOCALIZACAO=2 >  0.50
|   |   |   |--- class: 1
|   |--- TP_DEPENDENCIA=2 >  0.50
|   |   |--- IN_ENERGIA_GERADOR_FOSSIL=1.0 <= 0.50
|   |   |   |--- class: 1
|   |   |--- IN_ENERGIA_GERADOR_FOSSIL=1.0 >  0.50
|   |   |   |--- class: 0


> Claude faz a validação cruazada 5-fold e reporta acurácia média

Apliquei a validação cruzada 5-fold estratificada (StratifiedKFold, embaralhada, seed 42) nos 5562 exemplos, e a acurácia média do seu ID3 ficou em 0,8765 com max_depth=3.

Acurácia média (± desvio padrão entre os folds)

Modelo	max_depth=3	Sem limite
ID3 próprio	0,8765 ± 0,0032	0,8752 ± 0,0034
sklearn (entropy)	0,8767 ± 0,0035	0,8750 ± 0,0036
sklearn (gini)	0,8763 ± 0,0036	0,8750 ± 0,0036
Baseline (sempre a classe 0)	0,8190 ± 0,0004	0,8190 ± 0,0004

Acurácia do ID3 próprio em cada fold com max_depth=3: 0,8733, 0,8805, 0,8768, 0,8732 e 0,8786.

Como interpretar

Ganho real sobre o baseline: o ID3 supera em cerca de 5,8 pontos percentuais a estratégia de prever sempre a classe majoritária. Os folds variam pouco (cerca de 0,3 ponto), então o resultado é estável.
Empate com o sklearn: as diferenças entre os modelos (no máximo 0,0004 na média) são bem menores que o desvio entre folds, então são equivalentes em acurácia.
Limitar a profundidade não prejudica: a árvore completa não melhora (0,8752 contra 0,8765), e max_depth=3 é ligeiramente melhor e bem mais simples (13,8 folhas em média contra 44,4). É um bom argumento para manter a árvore de 3 níveis.
Tamanho das árvores: o ID3 multiway tem mais folhas que o sklearn nos dois cenários (13,8 contra 8,0 com max_depth=3). Isso reflete a divisão em um ramo por valor, já vista na tarefa 5.

Observações de método

Divisão estratificada: cada fold mantém a proporção de classes (cerca de 82% / 18%).
Domínio dos atributos: o ID3 recalcula o domínio dos atributos só com o treino de cada fold, sem usar informação do teste. Um valor nunca visto no treino cai na classe majoritária do nó.
Mesmo particionamento: o script usa os mesmos folds para todos os modelos, então a comparação é pareada.

Para rodar: python validacao_cruzada.py dados_etapa1.csv, com o script na mesma pasta do learn_decision_tree.py e do comparar_sklearn.py, que ele importa.


> Rodando a validação cruzada

No terminal:
python validacao_cruzada.py dados_etapa1.csv

Saída:
==================================================================
5-fold estratificado | max_depth=3
==================================================================
fold  ID3 próprio  sk-entropy   sk-gini  baseline
   1       0.8733      0.8733    0.8733    0.8185
   2       0.8805      0.8814    0.8814    0.8185
   3       0.8768      0.8768    0.8750    0.8192
   4       0.8732      0.8732    0.8732    0.8192
   5       0.8786      0.8786    0.8786    0.8192

Acurácia média ± desvio padrão (entre os folds):
  ID3 próprio                      0.8765 ± 0.0032
  sklearn (entropy)                0.8767 ± 0.0035
  sklearn (gini)                   0.8763 ± 0.0036
  baseline (classe majoritária)    0.8190 ± 0.0004

Folhas médias: ID3 próprio = 13.8 | sklearn = 8.0

==================================================================
5-fold estratificado | max_depth=None
==================================================================
fold  ID3 próprio  sk-entropy   sk-gini  baseline
   1       0.8706      0.8706    0.8706    0.8185
   2       0.8760      0.8760    0.8760    0.8185
   3       0.8768      0.8768    0.8768    0.8192
   4       0.8732      0.8723    0.8723    0.8192
   5       0.8795      0.8795    0.8795    0.8192

Acurácia média ± desvio padrão (entre os folds):
  ID3 próprio                      0.8752 ± 0.0034
  sklearn (entropy)                0.8750 ± 0.0036
  sklearn (gini)                   0.8750 ± 0.0036
  baseline (classe majoritária)    0.8190 ± 0.0004

Folhas médias: ID3 próprio = 44.4 | sklearn = 35.6







