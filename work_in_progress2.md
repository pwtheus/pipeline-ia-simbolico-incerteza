>>> Work in progress 2


> 1 - Gerando a base de dados (csv filtrado)  


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



