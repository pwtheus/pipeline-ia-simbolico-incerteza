>>> Work in progress 2


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

