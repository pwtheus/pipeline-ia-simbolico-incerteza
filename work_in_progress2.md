>>> Work in progress 2


> Gerando a base de dados (csv filtrado)  


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


