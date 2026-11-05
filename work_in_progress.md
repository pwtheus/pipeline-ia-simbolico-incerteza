>>>>> Work in progress




__________________________________________________________________________

>>> Etapa 1



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


> ChatGPT escreve sobre monotonicidade

Monotonicidade

A monotonicidade é uma propriedade importante em sistemas de raciocínio e aprendizagem de máquina. Um sistema é monotônico quando a inclusão de novas informações não invalida conclusões que já haviam sido obtidas anteriormente. Em outras palavras, uma vez que determinada conclusão é estabelecida, ela permanece válida mesmo quando novos exemplos ou informações são adicionados à base de conhecimento.

Entretanto, na aprendizagem de máquina, especialmente na construção de árvores de decisão, esse comportamento nem sempre ocorre. Um novo exemplo pode alterar a distribuição das classes de um conjunto de dados e, consequentemente, modificar os cálculos de entropia e ganho de informação. Isso pode levar o algoritmo a escolher outro atributo como raiz da árvore ou a modificar divisões que haviam sido estabelecidas anteriormente.

Por exemplo, em nosso trabalho, o atributo IN_LABORATORIO_INFORMATICA foi escolhido como raiz da árvore por apresentar o maior ganho de informação (0,1915). Essa escolha foi baseada na distribuição dos 5.562 exemplos disponíveis. Se novos registros fossem adicionados, alterando a distribuição das classes e dos valores dos atributos, outro atributo poderia passar a apresentar o maior ganho de informação. Assim, uma conclusão anterior sobre qual atributo deveria ocupar a raiz poderia deixar de ser válida.

Esse exemplo demonstra que o processo de aprendizagem não é necessariamente monotônico: novas informações podem modificar conclusões anteriores, exigindo que o modelo seja recalculado ou treinado novamente. Portanto, diferentemente de um sistema monotônico, no qual novas informações apenas acrescentam conhecimento, um algoritmo como o ID3 pode revisar suas decisões à medida que novos exemplos são incorporados.

Entendendo a relação com o trabalho
Há três pontos importantes que ajudam a compreender o texto:

1. A conclusão inicial: com os 5.562 registros, o atributo IN_LABORATORIO_INFORMATICA apresentou o maior ganho de informação (0,1915), sendo escolhido como raiz.

2. A chegada de novos exemplos: se novos registros forem adicionados, as proporções das classes podem mudar. Consequentemente, os valores de entropia e ganho de informação também podem mudar.

3. A revisão da conclusão: outro atributo pode passar a ter o maior ganho de informação, fazendo com que a escolha inicial da raiz seja substituída.

Uma observação conceitual: a monotonicidade não significa que um algoritmo nunca possa mudar suas previsões. Ela diz respeito à possibilidade de novas informações invalidarem conclusões anteriores. No caso do ID3, isso fica evidente porque suas decisões dependem dos exemplos disponíveis durante o treinamento.










_________________________________________________________________________

>>> Etapa 2


> 1 - Gerando a base de dados (csv filtrado) e tarefa1


Prompt:  
Quero que você gere o cvs (e também o script gerador), extraído a partir dos dados enviados, com as seguintes configurações:

Variável-alvo: Densidade de Informatização: (QT\_DESKTOP\_ALUNO + QT\_COMP\_PORTATIL\_ALUNO + QT\_TABLET\_ALUNO) / QT\_SALAS\_UTILIZADAS (faça o cálculo)

Variáveis:  
IN\_INTERNET\_ALUNOS  
IN\_LABORATORIO\_INFORMATICA  
IN\_INTERNET\_APRENDIZAGEM  
QT\_EQUIP\_TV  
QT\_EQUIP\_MULTIMIDIA  
IN\_EQUIP\_LOUSA\_DIGITAL

Mantenha no documento apenas o que for essencial para trabalhar minha base de dados para Treinamento.

Trate os problemas que você identificou antes.

Filtre o escopo para apenas as escolas do Estado do Amazonas.


Resposta do Claude:  
Gerei o CSV e o script. A base tem **5.526 escolas do Amazonas** e 7 colunas: `DENSIDADE\_INFORMATIZACAO` (alvo, já calculada) e as 6 preditoras, sem identificadores nem colunas extras.

**Como o script filtra a base nacional:**

| **Etapa** | **Escolas** |
| :-: | :-: |
| Base nacional | 214.192 |
| Estado do Amazonas (`SG\_UF == "AM"`) | 6.104 |
| Em atividade (`TP\_SITUACAO\_FUNCIONAMENTO == 1`) | 5.562 |
| Até 60 salas (remove outliers) | 5.545 |
| Sem valores ausentes (final) | **5.526** |

**Tratamentos aplicados:**

- O código 88888 (“não informado”) nas colunas `QT\_\*` virou ausente, e as linhas afetadas foram descartadas. 

- O alvo só é calculado quando as três colunas de dispositivos estão informadas. Isso evita tratar “não informado” como zero. 

- As flags `IN\_\*` ficaram como inteiros 0/1 e o alvo foi arredondado em 4 casas. 

- O script valida que não há ausentes, que o alvo não é negativo e que as flags só têm 0 ou 1. 

- Não cortei o topo da distribuição do alvo, como fiz na análise nacional. Com poucas escolas, isso removeria dados reais. 

**Um ponto importante sobre o Amazonas:** **62% das escolas têm densidade 0** (mediana 0, média 0,68, máximo 32,7, assimetria 5,3). É bem mais concentrado em zero do que no país (30%), então uma regressão linear direta vai ter dificuldade. Para o trabalho, considere:

- usar log(1 + y) como alvo, o que dá `np.log1p(y)` na hora de treinar; 

- comentar essa característica na análise, ou ampliar o escopo para a região Norte; 

- usar apenas as escolas com densidade maior que 0, se quiser modelar “quanto”, e não “se tem”. 

Para reproduzir: `python3 gerar\_base\_amazonas.py \<caminho\_do\_Tabela\_Escola\_2025\_V2.csv\> \[saida.csv\]`. O CSV usa vírgula como separador, ponto decimal e codificação UTF-8, então `pd.read\_csv` lê direto. Se for abrir no Excel em português, talvez precise importar com vírgula como delimitador.


Obs: O que foi realizado acima já responde à Tarefa 1: Escolher uma variável-alvo contínua (ex.: valor da internação, renda média, taxa de criminalidade).

No caso, a variável escolhida foi a Densidade de Informatização:
DENSIDADE_INFORMATIZACAO = (QT_DESKTOP_ALUNO + QT_COMP_PORTATIL_ALUNO + QT_TABLET_ALUNO) / QT_SALAS_UTILIZADAS


> Tarefa 2: Implementar regressão linear univariada (fórmula fechada).

Claude responde:
Implementei a regressão linear univariada com fórmula fechada no regressao_univariada.py, usando só NumPy/pandas (o scikit-learn entra apenas para conferir).

Fórmula implementada (modelo ŷ = θ0 + θ1·x):

θ1 = Σ(xᵢ − x̄)(yᵢ − ȳ) / Σ(xᵢ − x̄)²
θ0 = ȳ − θ1·x̄

O script divide a base em treino e teste (80/20, semente 42), ajusta um modelo para cada preditora e calcula MSE, RMSE e R². Ele também confere os coeficientes com o scikit-learn: a diferença fica na ordem de 1e-15, ou seja, a implementação está correta.

Escolhas que fiz sem você ter dito: usei a base do Norte e rodei todas as 6 preditoras. Para usar a base do Amazonas ou escolher uma variável, basta passar o CSV e --feature NOME. Com --log, o alvo vira log(1+y).

Resultados no teste (base do Norte):

Preditora	R² (alvo bruto)	R² (alvo em log)
IN_INTERNET_ALUNOS	0,147	0,256
IN_LABORATORIO_INFORMATICA	0,122	0,240
QT_EQUIP_MULTIMIDIA	0,077	0,128
IN_INTERNET_APRENDIZAGEM	0,076	0,138
IN_EQUIP_LOUSA_DIGITAL	0,057	0,070
QT_EQUIP_TV	0,050	0,097
A melhor preditora isolada é IN_INTERNET_ALUNOS. Com o alvo bruto, o modelo é ŷ = 0,2499 + 1,8105·x: escolas com internet para alunos têm, em média, 1,81 dispositivos a mais por sala.
O alvo em log melhora o R² em todos os casos, como já tinha aparecido antes.
Uma variável sozinha explica pouco, no máximo 26%. Isso é esperado e justifica passar depois para a regressão multivariada.
IN_INTERNET_ALUNOS e IN_LABORATORIO_INFORMATICA são binárias, então a “reta” liga só duas médias (x = 0 e x = 1). Para um gráfico mais ilustrativo, gerei o de QT_EQUIP_MULTIMIDIA, que é contínua.

Se o enunciado pedir uma variável específica ou o gradiente descendente como comparação (provavelmente a próxima tarefa), me avise que eu adapto.

Gráfico - Regressão Linear Univariada:
grafico_univariada.png


> Rodando o script da tarefa 2
python regressao_univariada.py dados_etapa2.csv

Saída:
Base: 5526 linhas | treino: 4421 | teste: 1105 | alvo: y (bruto)

                   feature  theta0  theta1  r_pearson  mse_treino  mse_teste  rmse_teste  r2_treino  r2_teste
        IN_INTERNET_ALUNOS  0.2776  2.2107     0.4748      2.5941     2.0311      1.4252     0.2254    0.2372
IN_LABORATORIO_INFORMATICA  0.2976  2.3636     0.4853      2.5603     2.0976      1.4483     0.2355    0.2122
               QT_EQUIP_TV  0.3985  0.2449     0.3430      2.9548     2.2886      1.5128     0.1177    0.1404
       QT_EQUIP_MULTIMIDIA  0.4741  0.2004     0.3031      3.0413     2.3041      1.5179     0.0919    0.1346
  IN_INTERNET_APRENDIZAGEM  0.2566  1.0219     0.2767      3.0925     2.4640      1.5697     0.0766    0.0746
    IN_EQUIP_LOUSA_DIGITAL  0.6718  1.5587     0.1128      3.3063     2.6182      1.6181     0.0127    0.0166

Modelo: ŷ = 0.2776 + 2.2107 · IN_INTERNET_ALUNOS   (R² teste = 0.2372)
Conferência com scikit-learn: |Δθ0| = 6.88e-15, |Δθ1| = 3.60e-14


> Tarefa 3: Implementar regressão multivariada com gradiente descendente do zero

Claude Responde:
Implementei a regressão linear multivariada com gradiente descendente do zero, só com NumPy, no regressao_multivariada_gd.py, usando o dados_etapa2.csv que você reenviou (5.526 escolas do Amazonas, alvo + 6 preditoras).

Como funciona (ŷ = Xθ, com coluna de 1s para o intercepto):

Custo: J(θ) = (1/2m)·Σ(ŷ − y)²
Gradiente: ∇J = (1/m)·Xᵀ(Xθ − y)
Atualização: θ ← θ − α·∇J, parando quando a variação do custo fica abaixo de 1e-10
As preditoras são padronizadas (z-score, com média e desvio do treino) para convergir bem. No fim, θ volta para a escala original, então os coeficientes são interpretáveis.
A divisão é 80/20 (semente 42). Com --log o alvo vira log(1+y).

Resultados com α = 0,1:

Alvo	Iterações	J inicial → final	R² treino	R² teste
bruto	163	1,92 → 1,14	0,316	0,323
log(1+y)	145	0,200 → 0,072	0,532	0,497

Validação: os θ do gradiente descendente batem com a equação normal (lstsq) com diferença de até ~1e-4. Essa diferença vem do critério de parada, não de erro de implementação. Treino e teste próximos indicam que não há overfitting.

Taxa de aprendizado (alvo em log):

α = 0,001: não converge em 2.000 iterações
α = 0,01: converge em 1.215 iterações
α = 0,1: converge em 145 iterações
α = 0,3: converge em 51 iterações
α ≥ 1,0: diverge (o script avisa com um erro)

Isso rende uma boa discussão para o relatório. O gráfico convergencia_gd.png mostra a queda do custo.

Coeficientes (alvo em log): IN_LABORATORIO_INFORMATICA (0,56) e IN_INTERNET_ALUNOS (0,42) têm o maior peso por unidade. QT_EQUIP_MULTIMIDIA quase não contribui (0,004), já que a informação está em grande parte nas outras variáveis.

Se a tarefa pedir algo adicional, como comparar com sklearn, usar SGDRegressor, versão estocástica ou regularização, posso incluir.


Gráfico: convergencia_gd.png


> Rodando o script da tarefa 3
python regressao_multivariada_gd.py dados_etapa2.csv

Saída:
Base: 5526 linhas | treino: 4421 | teste: 1105 | alvo: y (bruto)
alpha = 0.1 | iterações realizadas: 163 | J inicial = 1.9192 -> J final = 1.1447

                 parametro  theta_GD  theta_eq_normal  |diferenca|
                intercepto  0.014041         0.014034     0.000007
        IN_INTERNET_ALUNOS  1.093991         1.093884     0.000107
IN_LABORATORIO_INFORMATICA  1.376792         1.376916     0.000125
  IN_INTERNET_APRENDIZAGEM  0.291698         0.291724     0.000026
               QT_EQUIP_TV  0.079129         0.079132     0.000003
       QT_EQUIP_MULTIMIDIA  0.014025         0.014016     0.000009
    IN_EQUIP_LOUSA_DIGITAL  0.168362         0.168417     0.000055

Treino: MSE = 2.2893 | R² = 0.3164
Teste : MSE = 1.8030 | RMSE = 1.3428 | R² = 0.3228



> Tarefa 4: Aplicar regularização L2 (Ridge) e plotar a curva de coeficientes vs. λ.

Claude Responde:
Implementei a regressão Ridge (L2) do zero no regressao_ridge.py e gerei o gráfico de coeficientes vs. λ. O script importa funções do regressao_multivariada_gd.py, então deixe os dois arquivos na mesma pasta. Usei o dados_etapa2.csv com o alvo em log.

O que está implementado:

Custo: J(θ) = (1/2m)·Σ(ŷ−y)² + (λ/2m)·Σθⱼ². O intercepto θ0 não é penalizado.
Gradiente: ∇J = (1/m)·Xᵀ(Xθ−y) + (λ/m)·[0, θ₁…θₙ].
Também há a solução fechada, θ = (XᵀX + λI₀)⁻¹Xᵀy, usada para o gráfico e para validar o gradiente.
A validação confirma que o gradiente com L2 bate com a fechada (diferença ≤ 5e-7 para λ = 0, 10 e 1000).
O λ ótimo foi escolhido por validação cruzada em 5 dobras, usando só o treino.

Curva de coeficientes (gráfico à esquerda):

Para λ < ~100, os coeficientes ficam praticamente iguais aos do OLS.
A partir de ~10³ eles encolhem, e em 10⁵ tendem a zero. Os coeficientes nunca chegam a zero exatamente, o que distingue Ridge de Lasso.
QT_EQUIP_MULTIMIDIA e IN_EQUIP_LOUSA_DIGITAL primeiro sobem (λ ≈ 10³–10⁴) e depois caem. Isso é típico do Ridge: quando os coeficientes maiores encolhem, as variáveis correlacionadas ganham peso.
λ	MSE teste	R² teste
0 (OLS)	0,13559	0,4974
7,9 (ótimo CV)	0,13556	0,4975
1.000	0,13464	0,5009
10.000	0,16442	0,3906

Conclusão para o relatório: a regularização praticamente não ajuda aqui. A curva de CV (gráfico à direita) é plana até λ ≈ 10², então o “ótimo” 7,9 é só o ponto mais baixo de um platô. A base tem 4.421 linhas de treino e só 6 preditoras com baixa colinearidade (VIF < 2,5), ou seja, há pouco risco de overfitting. O Ridge só piora com λ muito grande (underfitting). O valor está em mostrar esse comportamento, não em melhorar o modelo.

O gráfico usa coeficientes das variáveis padronizadas, para que sejam comparáveis entre si.


Gráfico: ridge_coeficientes.png


> Rodando script da tarefa 4
python regressao_ridge.py dados_etapa2.csv

Saída:
Validação (GD com L2 vs. solução fechada), diferença máxima entre θ:
  λ =       0: 5.02e-07
  λ =      10: 5.78e-07
  λ =    1000: 3.38e-07

λ ótimo por CV (5 dobras): 77.43  (MSE CV = 2.32503)

         λ    MSE teste   R² teste
         0      1.80304     0.3228
     77.43      1.80064     0.3237  <- ótimo CV
       100      1.80000     0.3239
      1000      1.78942     0.3279
     1e+04      1.97160     0.2595

Coeficientes (variáveis padronizadas):
                 preditora  theta_OLS (λ=0)  theta_Ridge (λ=77.4)
        IN_INTERNET_ALUNOS          0.42991               0.42534
IN_LABORATORIO_INFORMATICA          0.51733               0.50911
  IN_INTERNET_APRENDIZAGEM          0.14457               0.14488
               QT_EQUIP_TV          0.20282               0.20205
       QT_EQUIP_MULTIMIDIA          0.03880               0.04356
    IN_EQUIP_LOUSA_DIGITAL          0.02231               0.02243

Gráfico salvo em ridge_coeficientes.png


> Tarefa 5: Implementar regressão logística (sigmoide + gradiente) para o problema de classificação da Etapa 1.

Claude Responde:
Implementei a regressão logística do zero (só NumPy) no regressao_logistica.py, usando o dados_etapa1.csv (5.562 escolas, 18,1% positivas). Também gerei a curva de custo.

Implementação:

Sigmoide σ(z) = 1/(1+e⁻ᶻ), numericamente estável.
Custo de entropia cruzada: J = −(1/m)·Σ[y·log ŷ + (1−y)·log(1−ŷ)].
Gradiente (1/m)·Xᵀ(σ(Xθ)−y) e atualização θ ← θ − α∇J.
As métricas (acurácia, precisão, recall, F1, matriz de confusão e AUC) também são calculadas do zero.
Pré-processamento: TP_LOCALIZACAO virou RURAL (0/1) e TP_DEPENDENCIA virou dummies (referência = municipal). O CO_ENTIDADE, que é um identificador, foi descartado.
Divisão 80/20 estratificada (semente 42).

Resultados no teste (limiar 0,5):

Métrica	Valor
Acurácia	0,884 (classe majoritária: 0,819)
Precisão	0,769
Recall	0,512
F1	0,615
AUC	0,836
Matriz [[TN, FP], [FN, TP]]	[[880, 31], [98, 103]]
Acurácia sozinha engana. Ganha só 6,5 pontos sobre o baseline, e o recall de 51% mostra que o modelo deixa passar metade das escolas com infraestrutura digital básica. Por isso reportei também precisão, recall e AUC.
Limiar: escolhendo o limiar que maximiza o F1 no treino (0,25), o teste vai a acurácia 0,893, recall 0,622 e F1 0,678.
Não há vazamento. O alvo não é função direta das preditoras (o máximo teórico com essas variáveis seria cerca de 88%), então o resultado é plausível.
Maior efeito: IN_LABORATORIO_INFORMATICA tem odds ratio de ~14. Ter laboratório multiplica por 14 as chances de a escola ter infraestrutura básica. Ser rural reduz as chances em ~63%.

Validação e um ponto a observar:

Conferi o resultado com LogisticRegression do scikit-learn sem regularização. A diferença máxima em θ é 1,8e-3, residual do critério de parada.
Na primeira versão a diferença era 1,2. A causa é que DEP_FEDERAL tem só 18 escolas no treino (17 positivas), o que quase separa os dados e deixa esse coeficiente convergindo devagar. Por isso os padrões ficaram em α = 1,0 e até 100.000 iterações, com parada antecipada (converge em ~37 mil, 5 s).
O coeficiente de DEP_FEDERAL (2,76) é pouco confiável, pela amostra pequena. Vale mencionar isso no relatório.


Pedi para o Claude reiterar o gráfico:
O gráfico estava correto, mas pouco informativo, então refiz.

Por que estava correto: o custo começa em 0,693 (= ln 2, esperado com θ = 0, onde toda previsão é 0,5) e cai de forma monótona até ~0,3243. Isso confirma que o gradiente está certo e que α = 1,0 não diverge.

Por que era ruim: quase toda a queda acontece nas primeiras ~100 iterações. Numa escala linear de 37.000 iterações, isso virava uma linha vertical seguida de uma reta horizontal, e a cauda lenta (causada por DEP_FEDERAL) não aparecia.

O que mudou: agora são dois painéis.

Esquerda: as 300 primeiras iterações, onde o custo cai de 0,69 para ~0,33.
Direita: o treino completo com eixo x em escala log e uma linha tracejada no J final (0,3243). Dá para ver a descida rápida até ~10² e o platô depois.

O restante do script não mudou, só o bloco do gráfico. Para o relatório, o painel da direita é o mais útil, porque mostra tanto a convergência quanto o platô.

Gráfico: custo_logistica.png


> Rodando o script da tarefa 5
python regressao_logistica.py dados_etapa1.csv

Saída:
Base: 5562 linhas | treino: 4450 | teste: 1112 | positivos: 18.1% (estratificado)
alpha = 1.0 | iterações: 37421 | J: 0.6931 -> 0.3243

                 parametro   theta  odds_ratio
                intercepto -1.7718      0.1700
IN_LABORATORIO_INFORMATICA  2.6363     13.9620
             IN_BIBLIOTECA  0.4156      1.5153
 IN_ENERGIA_GERADOR_FOSSIL -0.3452      0.7081
                     RURAL -0.9921      0.3708
              DEP_ESTADUAL -0.9279      0.3954
               DEP_FEDERAL  2.7637     15.8579
               DEP_PRIVADA  0.2171      1.2425

Conferência com scikit-learn: diferença máxima em θ = 1.78e-03

Treino (limiar 0.5): acurácia = 0.8676 | precisão = 0.6641 | recall = 0.5447 | F1 = 0.5985 | AUC = 0.8474
  matriz de confusão [[TN, FP], [FN, TP]] = [[3422, 222], [367, 439]]

Teste (limiar 0.5): acurácia = 0.8840 | precisão = 0.7687 | recall = 0.5124 | F1 = 0.6149 | AUC = 0.8358
  matriz de confusão [[TN, FP], [FN, TP]] = [[880, 31], [98, 103]]

Referência: classificar tudo como classe majoritária = 0.8192 de acurácia

Limiar que maximiza F1 no treino: 0.25 -> teste: acurácia = 0.8930 | precisão = 0.7440 | recall = 0.6219 | F1 = 0.6775


> Tarefa 6: Comparar com LinearRegression e LogisticRegression do scikit-learn.

Claude Responde:
Comparei as suas implementações com o scikit-learn no comparar_sklearn.py. O resultado é que elas coincidem. O script importa os três scripts anteriores (regressao_multivariada_gd.py, regressao_ridge.py e regressao_logistica.py), então mantenha todos na mesma pasta. Os testes usam as mesmas divisões treino/teste do resto do trabalho. O .txt traz a saída completa.

A) Regressão linear (GD do zero × LinearRegression, dados_etapa2.csv, alvo em log)

	GD do zero	sklearn
Coeficientes	idênticos até a 6ª casa	
MSE teste	0,135588	0,135588
R² teste	0,497421	0,497421
Diferença máx. nas previsões	5e-7	

O resultado é o mesmo com o alvo bruto (R² 0,3228 nos dois). O GD levou 277 iterações, com tolerância 1e-15.

B) Ridge (do zero × Ridge)

Usando λ = alpha do sklearn, a convenção do custo coincide. Para λ = 0,1, 10 e 1000, a diferença nos θ é de ~1e-7 para o GD e de ~1e-14 para a solução fechada.
O MSE de teste é idêntico nos dois.

C) Regressão logística (do zero × LogisticRegression, dados_etapa1.csv)

Modelo	Acurácia	F1	AUC	Custo teste
Do zero	0,8840	0,6149	0,8358	0,31765
sklearn sem regularização	0,8840	0,6149	0,8358	0,31765
sklearn padrão (L2, C=1)	0,8840	0,6149	0,8358	0,31753
Contra o sklearn sem regularização (C=1e12), os coeficientes diferem em até 2e-4, e as classes previstas são 100% iguais (probabilidades diferem em até 3e-6).
O LogisticRegression com configuração padrão aplica regularização L2 (C = 1), então seus coeficientes diferem dos do seu modelo. O mais visível é DEP_FEDERAL, que cai de 2,77 para 1,50, por causa da amostra pequena (18 escolas). As métricas no teste ficam iguais e o custo é um pouco menor (0,31753 contra 0,31765). Vale destacar isso no relatório, pois muita gente compara com o padrão e acha que a implementação está errada.

Tempo:

Linear: praticamente igual (0,015 s contra 0,009–0,06 s).
Logística: a sua leva ~5,6 s (50 mil iterações) contra 0,025 s do sklearn, que usa L-BFGS e converge em poucas iterações. É um bom ponto de discussão: o gradiente descendente puro é correto, mas mais lento.


> Rodando script da tarefa 6:
python comparar_sklearn.py dados_etapa2.csv dados_etapa1.csv

Saída:
==============================================================================
A) REGRESSÃO LINEAR (alvo: y bruto) — 4421 treino / 1105 teste
==============================================================================
                 parametro  GD do zero  sklearn    |dif|
                intercepto    0.014034 0.014034 0.000000
        IN_INTERNET_ALUNOS    1.093885 1.093884 0.000001
IN_LABORATORIO_INFORMATICA    1.376916 1.376916 0.000001
  IN_INTERNET_APRENDIZAGEM    0.291724 0.291724 0.000000
               QT_EQUIP_TV    0.079132 0.079132 0.000000
       QT_EQUIP_MULTIMIDIA    0.014016 0.014016 0.000000
    IN_EQUIP_LOUSA_DIGITAL    0.168417 0.168417 0.000000

                          GD do zero       sklearn
MSE teste                   1.803040      1.803040
R² teste                    0.322800      0.322800
tempo (s)                     0.0333        0.1229   (GD: 283 iterações)
Máx. diferença entre previsões: 9.14e-07

------------------------------------------------------------------------------
B) RIDGE (L2) — mesmo λ nos dois (J = (1/2m)Σerr² + (λ/2m)Σθ²  ≡  sklearn alpha = λ)
------------------------------------------------------------------------------
        lam  |GD - sklearn|  |fechada - sklearn|  MSE teste (meu)  MSE teste (sk)
   0.100000       2.353e-07            2.439e-14         1.803037        1.803037
  10.000000       2.381e-07            2.463e-14         1.802712        1.802712
1000.000000       1.724e-07            1.724e-14         1.789415        1.789415

==============================================================================
C) REGRESSÃO LOGÍSTICA — 4450 treino / 1112 teste (limiar 0,5)
==============================================================================
                 parametro   do zero  sk sem reg. (C=1e12)  sk padrão (L2, C=1)  |dif| vs sem reg.
                intercepto -1.771801             -1.771802            -1.770276           0.000001
IN_LABORATORIO_INFORMATICA  2.636338              2.636337             2.597703           0.000000
             IN_BIBLIOTECA  0.415597              0.415597             0.430738           0.000000
 IN_ENERGIA_GERADOR_FOSSIL -0.345235             -0.345237            -0.320831           0.000003
                     RURAL -0.992042             -0.992041            -0.984656           0.000001
              DEP_ESTADUAL -0.927917             -0.927916            -0.900621           0.000001
               DEP_FEDERAL  2.765255              2.765447             1.501139           0.000192
               DEP_PRIVADA  0.217120              0.217121             0.216932           0.000001

             modelo  acurácia       F1      AUC  custo teste  tempo (s)
            do zero  0.883993 0.614925 0.835843     0.317649  21.289334
   sklearn sem reg.  0.883993 0.614925 0.835843     0.317649   0.079622
sklearn padrão (L2)  0.883993 0.614925 0.835843     0.317532   0.017031

Do zero: 49972 iterações.
Classes previstas iguais (do zero vs sklearn sem reg.): 100.00%
Máx. diferença nas probabilidades (do zero vs sklearn sem reg.): 3.22e-06








_________________________________________________________________________

>>> Etapa 5 - Redes Bayesianas

Claude:

Preparação dos dados

Etapa 1 ou Etapa 2? Redes Bayesianas clássicas trabalham com tabelas de probabilidade condicional (CPTs), então variáveis categóricas ou binárias são o ideal.

Etapa 1 (recomendada): todas as colunas já são discretas: IN_LABORATORIO_INFORMATICA, IN_BIBLIOTECA, TP_LOCALIZACAO, TP_DEPENDENCIA, IN_ENERGIA_GERADOR_FOSSIL e o alvo infra_digital_basica. São 6 variáveis, exatamente o limite do enunciado, sem transformação.


> 1. Desenhar o DAG com 4–6 nós (justificar cada aresta).

DAG (6 nós, 8 arestas)
                 TP_LOCALIZACAO
                  /          \
                 ▼            ▼
      TP_DEPENDENCIA     IN_ENERGIA_GERADOR_FOSSIL
        |         \          /            |
        ▼          ▼        ▼             |
  IN_BIBLIOTECA    IN_LABORATORIO_INFORMATICA
        \                |               /
         ▼               ▼              ▼
                infra_digital_basica
Justificativa das arestas
#	Aresta	Justificativa
1	TP_LOCALIZACAO → TP_DEPENDENCIA	Escolas rurais são majoritariamente municipais, e a rede privada se concentra em áreas urbanas.
2	TP_LOCALIZACAO → IN_ENERGIA_GERADOR_FOSSIL	Em áreas rurais e ribeirinhas do AM, a rede elétrica é limitada e muitas escolas dependem de gerador a combustível fóssil.
3	TP_DEPENDENCIA → IN_LABORATORIO_INFORMATICA	A rede (federal, estadual, municipal ou privada) define orçamento e política de investimento em laboratórios.
4	TP_DEPENDENCIA → IN_BIBLIOTECA	O investimento em infraestrutura física varia por rede.
5	IN_ENERGIA_GERADOR_FOSSIL → IN_LABORATORIO_INFORMATICA	Equipamentos de informática exigem energia confiável. Depender de gerador dificulta instalar e manter um laboratório.
6	IN_ENERGIA_GERADOR_FOSSIL → infra_digital_basica	Internet para alunos requer energia estável. O efeito é direto, além do que passa pelo laboratório (aresta 5 seguida da 7).
7	IN_LABORATORIO_INFORMATICA → infra_digital_basica	Escolas com laboratório tendem a ter desktops e conexão disponíveis aos alunos. É uma associação forte, mas não definicional.
8	IN_BIBLIOTECA → infra_digital_basica	Proxy de investimento geral em infraestrutura escolar. É a aresta mais fraca, e dá para testá-la com qui-quadrado ou informação mútua.
Decisões de modelagem
Acíclico: a ordem topológica LOCALIZACAO → DEPENDENCIA, GERADOR → BIBLIOTECA, LABORATORIO → infra_digital_basica existe.
Tamanho das CPTs: o alvo tem 3 pais binários, ou seja, 8 configurações, o que é seguro com 5.562 linhas. Não adicionei TP_DEPENDENCIA → infra_digital_basica nem TP_LOCALIZACAO → infra_digital_basica para evitar CPTs esparsas. Parte desse efeito já passa pelo laboratório e pela energia. Se quiser incluir uma delas, justifique no relatório e confira se há combinações com poucas observações.
Linguagem na justificativa: descreva as arestas como dependências plausíveis do domínio, não como causalidade comprovada.


Imagem: dag_rede_bayesiana.pgn


> 2. Estimar as CPTs a partir dos dados.

As 6 CPTs estimadas com os 5.562 registros (sem valores faltantes), usando suavização de Laplace com α = 1, estão abaixo. Conferi os números por uma segunda contagem independente e batem. Para os nós binários mostro só P(X = 1), porque P(X = 0) é o complemento. Os arquivos completos estão no zip.

1. P(TP_LOCALIZACAO)

1 (urbana)	2 (rural)
0,3409	0,6591

2. P(TP_DEPENDENCIA | TP_LOCALIZACAO) (N: 1.896 urbanas, 3.666 rurais)

Dependência	Urbana	Rural
1 (federal)	0,0089	0,0011
2 (estadual)	0,3095	0,0469
3 (municipal)	0,4705	0,9507
4 (privada)	0,2111	0,0014

3. P(IN_ENERGIA_GERADOR_FOSSIL = 1 | TP_LOCALIZACAO)

Urbana	Rural
0,0116	0,1838

4. P(IN_BIBLIOTECA = 1 | TP_DEPENDENCIA)

Dependência	1	2	3	4
P(biblioteca = 1)	0,8571	0,6961	0,1285	0,6133
N	19	758	4.381	404

5. P(IN_LABORATORIO_INFORMATICA = 1 | TP_DEPENDENCIA, GERADOR)

Dependência	Gerador = 0	N	Gerador = 1	N
1	0,8750	14	0,7143	5
2	0,5412	739	0,0952	19
3	0,1075	3.727	0,0213	654
4	0,3000	388	0,5000	16

6. P(infra_digital_basica = 1 | GERADOR, LABORATÓRIO, BIBLIOTECA)

Gerador	Lab	Biblioteca	P(Y = 1)	N
0	0	0	0,0696	3.358
0	0	1	0,1784	581
0	1	0	0,6618	205
0	1	1	0,6639	724
1	0	0	0,0376	636
1	0	1	0,3235	32
1	1	0	0,7000	8
1	1	1	0,8000	18
O que os dados dizem sobre as arestas
Arestas bem sustentadas:
Localização → Dependência: 95% das escolas rurais são municipais, contra 47% das urbanas.
Localização → Gerador: 18,4% das rurais usam gerador, contra 1,2% das urbanas.
Dependência → Biblioteca: só 12,9% das municipais têm biblioteca, contra 61% a 86% nas demais redes.
Laboratório → alvo: ter laboratório eleva P(Y = 1) de cerca de 7% para cerca de 66%.
Gerador → Laboratório: nas redes estadual e municipal, o gerador reduz muito a chance de laboratório (54% para 9,5% e 10,8% para 2,1%).
Aresta 8 (Biblioteca → alvo) é a mais fraca: com laboratório, a biblioteca não muda nada (0,662 contra 0,664). Só pesa sem laboratório (0,070 para 0,178). Vale mencionar isso no relatório.
Aresta 6 (Gerador → alvo) é incerta: o efeito direto muda de sinal entre as configurações e as que têm gerador = 1 são pequenas (N entre 8 e 32).
Cuidados com amostras pequenas
Configurações com menos de 30 observações: dependência 1 (N = 19, incluindo 14 e 5 na tabela 5), as combinações dependência 2 com gerador, dependência 4 com gerador na tabela 5, e as duas últimas linhas com N pequeno nas tabelas 6 (N = 8, 18 e 32).
Nas linhas com N = 8 e N = 18 da tabela 6, a suavização com α = 1 puxa o valor para 0,5. A proporção bruta é 0,75 e 0,83, contra 0,70 e 0,80 suavizados. Cite o N ao lado dessas linhas no relatório.
Cerca de 18,1% das escolas têm infra_digital_basica = 1, e essa proporção será a referência para comparar as consultas de inferência.


Script: estimar_cpts.py

> 3. Implementar inferência por enumeração do zero.

Claude:

A inferência por enumeração está pronta no arquivo inferencia_enumeracao.py, em Python puro (o pandas só lê o CSV). Ele roda sobre as CPTs que estimamos e já traz um autoteste e consultas de exemplo.

Como funciona
consulta(rede, X, evidencia) implementa o ENUMERATION-ASK. Para cada valor de X, ele soma a probabilidade sobre todas as variáveis ocultas e normaliza no final.
enumerar_todas é o ENUMERATE-ALL recursivo. Ele percorre as variáveis em ordem topológica, e a ordem é verificada pelo script. Se a variável está na evidência, multiplica pelo seu fator. Se não, soma sobre todos os valores dela.
Evidência impossível: o script avisa quando a evidência tem probabilidade zero no modelo, ou quando o valor está fora do domínio.

Para consultar outra coisa, use a sintaxe "X | evidencia=valor, ...", por exemplo:

python inferencia_enumeracao.py dados_etapa1.csv "infra_digital_basica | TP_LOCALIZACAO=2, IN_LABORATORIO_INFORMATICA=1"
Validação

O autoteste confirma duas coisas. A distribuição conjunta soma 1. E a enumeração coincide, com diferença máxima de 2,2e-16, com um segundo método (marginalizar a conjunta completa de 128 combinações) em 76 consultas com zero ou uma evidência.

Resultados das consultas de exemplo

O script compara a rede com a frequência empírica nos dados. Algumas consultas:

Consulta	Rede	Empírico
P(Y=1)	0,189	0,181
P(Y=1 | Lab=1)	0,664	0,668
P(Y=1 | Rural)	0,148	0,078
P(Y=1 | Rural, Municipal)	0,135	0,076
P(Rural | Y=1)	0,518	0,284
O ponto a decidir no relatório

Quando a evidência envolve laboratório, gerador ou biblioteca, a rede acompanha os dados. Quando envolve a localização, ela erra bastante: escolas rurais têm 7,8% de infraestrutura básica nos dados, mas a rede prevê 14,8%. A causa é que o DAG não tem aresta direta TP_LOCALIZACAO → infra_digital_basica. A localização só chega ao alvo por laboratório, gerador e biblioteca, e esses três não explicam toda a diferença (provavelmente há também uma diferença de acesso à internet entre rural e urbano).

Eu havia deixado essa aresta de fora para manter a CPT do alvo pequena. Os resultados mostram que ela faz falta.

Há dois caminhos:

Manter o DAG e apresentar essa divergência como limitação do modelo, o que é honesto e válido para a tarefa.
Adicionar a aresta TP_LOCALIZACAO → infra_digital_basica. A CPT do alvo passa a ter 16 configurações e algumas ficam esparsas, mas a rede deve se aproximar bem mais dos dados.


Script: inferencia_enumeracao.py



> 4. Comparar com pgmpy (VariableElimination).

O que o script faz

Ele compara as duas coisas separadamente:

A) Mesmas CPTs nas duas implementações. Monta a rede no pgmpy com as CPTs que estimamos e roda VariableElimination nas 9 consultas de exemplo. Isso isola a diferença entre o seu algoritmo de enumeração e o de eliminação de variáveis.
B) CPTs estimadas pelo próprio pgmpy. Usa BayesianEstimator com prior_type="K2", que equivale a pseudo-contagem 1, ou seja, à nossa suavização de Laplace com α = 1. Compara entrada por entrada com as nossas CPTs e depois repete as consultas.

Como os dois métodos são exatos, o esperado é diferença da ordem de 1e-15. O script para com erro se passar de 1e-9, o que apontaria um problema de implementação ou de parâmetros.


Para o relatório, vale lembrar que enumeração percorre todas as combinações das variáveis ocultas (custo exponencial), enquanto a eliminação de variáveis reaproveita somas intermediárias. Nesta rede de 6 nós a diferença de tempo é irrelevante, mas a teoria é um bom ponto de comparação.


Script: comparar_pgmpy.py


> 5. Aplicar d-separação para verificar 2 independências condicionais.

Escolhi duas independências que o DAG implica, provei cada uma por d-separação e testei nos dados. Os testes rejeitam as duas, uma de forma clara e outra de forma leve, e isso é um resultado importante para o relatório.

Implementei a d-separação de dois jeitos, listando os caminhos e bloqueando cada um, e pelo grafo ancestral moralizado. Os dois algoritmos concordaram em todos os casos.

Independência 1: BIBLIOTECA ⟂ LABORATÓRIO | DEPENDÊNCIA
O único caminho sem colisor é BIB ← DEP → LAB, uma bifurcação, bloqueada ao observar DEP.
O caminho BIB ← DEP ← LOC → GER → LAB também passa por DEP e fica bloqueado.
Os demais passam pelo colisor Y (que não é observado), então estão bloqueados.
Conclusão: BIB e LAB são d-separados dado DEP.
Contraste: se observarmos também Y, o colisor abre os caminhos BIB → Y ← LAB e BIB → Y ← GER → LAB, e as variáveis voltam a ser d-conectadas. É o efeito de “explicar” a causa, o explaining away.
Independência 2: DEPENDÊNCIA ⟂ GERADOR | LOCALIZAÇÃO
O único caminho sem colisor é DEP ← LOC → GER, bloqueado ao observar LOC.
Os outros passam pelos colisores LAB ou Y, que não são observados.
Conclusão: DEP e GER são d-separados dado LOC.
Contraste: observar LAB abre DEP → LAB ← GER.
O que os dados mostram

Usei qui-quadrado estratificado pelo conjunto condicionante e a informação mútua condicional I(X;Y|Z), em bits.

Teste	Dependência marginal	Dado Z: qui² (gl)	p	I(X;Y|Z)
BIB ⟂ LAB | DEP	0,2009 bits	1286,6 (4)	≈ 3·10⁻²⁷⁷	0,119 bits
DEP ⟂ GER | LOC	0,0207 bits	181,1 (6)	≈ 2·10⁻³⁶	0,010 bits
Extra: Y ⟂ LOC | {GER, LAB, BIB}	—	75,7 (8)	≈ 4·10⁻¹³	0,009 bits

Com 5.562 escolas, o teste detecta até desvios pequenos, então olhe o tamanho do efeito junto com o p-valor:

Independência 1 é claramente violada. Condicionar em DEP remove só cerca de 40% da dependência. Dentro de cada rede, escolas com biblioteca tendem a ter laboratório, provavelmente por uma causa comum não modelada, como o porte ou o investimento geral da escola.
Independência 2 é violada de leve. Resta cerca de metade da dependência marginal, mas num valor pequeno (0,010 bits). Os estratos rurais das redes federal e privada têm poucas escolas, então o qui-quadrado é menos confiável ali.
O diagnóstico extra confirma o que vimos na inferência: falta a aresta TP_LOCALIZACAO → infra_digital_basica, e o teste a rejeita.

Não dá para usar os contrastes com colisor (observar Y ou LAB) para “confirmar” o explaining away nos dados, porque a dependência já existe sem eles. Eles valem como demonstração da teoria.

Como escrever no relatório

Você pode apresentar as duas independências como implicações do DAG e discutir os resultados como limitação do modelo. Ou posso refazer o DAG com duas arestas a mais: BIB–LAB (a direção é uma escolha de modelagem) e LOC → Y. Nesse caso reestimo as CPTs, refaço a inferência e repito os testes para você comparar as duas versões. Posso também testar todas as independências que o DAG implica e listá-las por tamanho do efeito.


Script: d_separacao.py


> 6. Calcular quantos parâmetros a rede economiza em relação à conjunta.

A rede precisa de 29 parâmetros livres contra 127 da distribuição conjunta, uma economia de 98 parâmetros (77,2%). Conferi a conta por código e bate com o cálculo à mão.

Conjunta

As cardinalidades são LOC = 2, DEP = 4, GER = 2, BIB = 2, LAB = 2 e Y = 2.

Número de células: 2 × 4 × 2 × 2 × 2 × 2 = 128. Como as probabilidades somam 1, os parâmetros livres são 128 − 1 = 127.

Rede

Cada nó precisa de (|X| − 1) × (produto das cardinalidades dos pais) parâmetros:

Nó	Pais	Cálculo	Parâmetros
TP_LOCALIZACAO	—	(2−1) · 1	1
TP_DEPENDENCIA	LOC	(4−1) · 2	6
IN_ENERGIA_GERADOR_FOSSIL	LOC	(2−1) · 2	2
IN_BIBLIOTECA	DEP	(2−1) · 4	4
IN_LABORATORIO_INFORMATICA	DEP, GER	(2−1) · 4 · 2	8
infra_digital_basica	GER, LAB, BIB	(2−1) · 2 · 2 · 2	8
Total			29

Contando as entradas de tabela em vez dos parâmetros livres, são 54 contra 128 (as CPTs que mostrei antes).

Economia: 127 − 29 = 98 parâmetros, ou seja, a rede usa 22,8% do tamanho da conjunta.

Efeito prático nos dados
Só 64 das 128 células da conjunta aparecem nos dados, e 19 delas têm menos de 5 escolas.
Há cerca de 44 escolas por parâmetro livre na conjunta, contra cerca de 192 na rede, então as estimativas da rede são bem mais estáveis.
A economia vem das independências condicionais que o DAG assume (por exemplo, BIB e LAB independentes dado DEP). Como mostrei na tarefa 5, algumas delas valem só de forma aproximada nos dados. Ou seja, a economia tem o preço de um pequeno viés.
Se você adotar as arestas sugeridas
Variante	Parâmetros	Economia
DAG atual	29	98
+ LOC → Y	37	90
+ BIB → LAB	37	90
+ as duas	45	82

Mesmo com as duas arestas extras, a rede continua usando menos de 36% dos parâmetros da conjunta.

