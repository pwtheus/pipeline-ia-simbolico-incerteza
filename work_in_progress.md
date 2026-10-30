\>\>\> Work in progress


\> Claude implementa a LEARN-DECISION-TREE


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



