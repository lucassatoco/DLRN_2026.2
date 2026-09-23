---
exercise: perceptron
ai_use: "Claude (Claude Code) para revisao do codigo, elaboracao dos textos e montagem do GitHub Pages."
---

# Exercise 2 — Perceptron

Neste exercício implementei o perceptron do zero e testei em dois casos: um em que as classes são
linearmente separáveis (Exercise 1) e outro em que elas se sobrepõem (Exercise 2). A ideia é ver na prática
quando o algoritmo converge e o que acontece quando ele não tem como convergir, que é onde entra o *pocket*.

**Como fiz.**

- A implementação está em [`code/perceptron.py`](code/perceptron.py): função degrau, previsão, regra de
  atualização e o laço de treino. Usei só NumPy para o modelo, `pandas` para montar tabelas e `matplotlib`
  para os gráficos. Os dois exercícios importam a mesma função `train_perceptron`, então o Exercise 2 usa
  exatamente o mesmo código do Exercise 1, sem mudar nada.
- Cada script começa com `rng = np.random.default_rng(42)` e todos os sorteios (dados, embaralhamento e pesos
  iniciais) saem desse gerador. Rodando os scripts de novo, as figuras e os números saem iguais.
- Depois de gerar os dados eu embaralho a ordem das amostras uma vez só, e essa ordem fica fixa durante o
  treino. Se não embaralhar, o perceptron vê 1000 pontos da classe 0 e depois 1000 da classe 1. Manter a ordem
  fixa também ajudou nas comparações do item D, porque quando eu mudo o η e mantenho o mesmo $\mathbf{w}_0$, a
  única coisa diferente entre as execuções é o η.
- Usei os valores do enunciado: $\mathbf{w}_0 \sim \mathcal{N}(0,\ 0.01)$ (desvio 0.01), $b_0 = 0$ e
  $\eta = 0.01$. O treino para quando passa uma época inteira sem nenhuma atualização ou quando chega em 100
  épocas. A acurácia é calculada no dataset inteiro no fim de cada época.
- O pocket é checado depois de **cada atualização**, como o enunciado pede: se a acurácia no dataset inteiro
  for maior que a melhor que eu já tinha visto, guardo uma cópia de $(\mathbf{w}, b)$. Ele não interfere no
  treino, só guarda os melhores pesos.

**Dificuldades.** A primeira foi o tempo de execução. Checar o pocket depois de cada atualização significa
calcular a acurácia nos 2000 pontos a cada erro, e no Exercise 2 são uns 700 erros por época (umas 70 mil
avaliações em 100 épocas). Resolvi calculando a acurácia de forma vetorizada (`X @ w + b` para todos os pontos
de uma vez), e com isso roda em poucos segundos. O laço de atualização continuou sendo amostra por amostra,
que é como o perceptron funciona. A segunda foi entender o resultado do Exercise 2: a acurácia final depende
de em que momento o treino para, então só o número não explica muita coisa. Por isso, na análise, eu também
olhei onde a fronteira ficou (ângulo, deslocamento e quantos pontos ela manda para cada classe).

Também deixei tudo em um notebook, com as saídas: [`code/perceptron_notebook.ipynb`](code/perceptron_notebook.ipynb).
Os números dele são os mesmos deste relatório.

---

## Exercise 1

**Separable data.** Este é o script do exercício. A implementação do perceptron que ele usa está no item B.

```python
--8<-- "docs/exercises/perceptron/code/ex1_separable.py"
```

### A — Generate the data

Gerei 1000 amostras por classe com $\mathcal{N}(\mu, \Sigma)$ e $\Sigma = 0.5\,I$: a classe 0 com
$\mu_0 = (1.5,\ 1.5)$ e a classe 1 com $\mu_1 = (5,\ 5)$. As médias das amostras ficaram (1.450, 1.473) e
(5.010, 5.013), bem perto das médias teóricas. A distância entre os centros é $3.5\sqrt 2 \approx 4.95$, e o
desvio em cada eixo é $\sqrt{0.5} \approx 0.71$. Ou seja, cada centro fica a uns 3.5 desvios da reta que
divide os dois, e dá para ver na figura que as nuvens não se encostam.

![Figure 1](figures/fig1.png)

### B — Implement the perceptron

O modelo e a regra de atualização que implementei:

$$\hat y = \operatorname{step}(\mathbf{w}\cdot\mathbf{x} + b), \qquad
\operatorname{step}(z) = \begin{cases}1 & z \ge 0\\ 0 & z < 0\end{cases}$$

$$\mathbf{w} \leftarrow \mathbf{w} + \eta\,(y - \hat y)\,\mathbf{x}, \qquad b \leftarrow b + \eta\,(y - \hat y)$$

Como $y - \hat y$ só pode ser $-1$, $0$ ou $+1$, os pesos só mudam quando o modelo erra. Se o ponto é da
classe 1 e o modelo previu 0, soma $\eta\mathbf{x}$ em $\mathbf{w}$; se é da classe 0 e previu 1, subtrai.

```python
--8<-- "docs/exercises/perceptron/code/perceptron.py"
```

### C — Train and measure

Treinei com $\eta = 0.01$, e o $\mathbf{w}_0$ sorteado foi $[0.00991,\ -0.00827]$. O treino terminou em
**2 épocas**: na primeira teve 48 atualizações e no fim dela todos os pontos já estavam certos; na segunda não
teve nenhum erro, e por isso o treino parou.

| | valor |
|---|---|
| $\mathbf{w}$ final | [0.03189, 0.02874] |
| $b$ final | −0.20 |
| épocas | 2 (48 atualizações, depois 0) |
| acurácia | **100%** |

A reta $0.03189\,x_1 + 0.02874\,x_2 - 0.20 = 0$ cruza a diagonal em $x_1 = x_2 \approx 3.30$, perto do ponto
médio entre os centros (3.25, 3.25), e o vetor $\mathbf{w}$ aponta quase na direção $(1, 1)$. Nenhum ponto
ficou mal classificado.

![Figure 2](figures/fig2.png)

![Figure 3](figures/fig3.png)

### D — Analysis

**1. Por que converge tão rápido?** A regra só atualiza quando erra, e toda atualização melhora a saída
naquele ponto: se era um falso negativo, $\mathbf{w}\cdot\mathbf{x} + b$ aumenta
$\eta(\lVert\mathbf{x}\rVert^2 + 1)$; se era um falso positivo, diminui o mesmo tanto. O teorema de
convergência do perceptron (Novikoff) diz que, se existe uma reta que separa os dados com margem $\gamma$, o
número total de erros é no máximo $(R/\gamma)^2$, onde $R$ é o maior $\lVert\mathbf{x}\rVert$. Aqui a margem
é grande em relação a $R$, porque as nuvens estão bem longe da reta que as separa, então esse limite é
pequeno, e de fato foram só 48 erros. Além disso, como o $\mathbf{w}_0$ é muito pequeno, logo nas primeiras
atualizações o $\mathbf{w}$ vira basicamente uma soma de pontos da classe 1 menos pontos da classe 0, e isso
aponta mais ou menos para $\mu_1 - \mu_0 \propto (1, 1)$, que já é uma boa direção.

**2. Rodando de novo com $\eta = 1.0$** (mesmo $\mathbf{w}_0$ e mesma ordem): também deu **2 épocas e 100% de
acurácia**, mas com 25 atualizações na primeira época em vez de 48. Os pesos finais foram
$\mathbf{w} = [1.717,\ 1.665]$ e $b = -11$.

| | $\eta = 0.01$ | $\eta = 1.0$ |
|---|---|---|
| $\mathbf{w}/\lVert\mathbf{w}\rVert$ | [0.7429, 0.6694] | [0.7181, 0.6960] |
| $\lVert\mathbf{w}\rVert$ | 0.043 | 2.392 |
| atualizações | 48, 0 | 25, 0 |

Os pesos ficaram uns 56 vezes maiores, mas a direção é quase a mesma: o cosseno entre os dois vetores
normalizados deu 0.9993, um ângulo de 2.1°. Então o η não muda muito a solução, ele muda o tamanho do passo
**comparado com o $\mathbf{w}_0$**. A fronteira $\mathbf{w}\cdot\mathbf{x} + b = 0$ continua a mesma se eu
multiplicar $(\mathbf{w}, b)$ por um número positivo, e a regra só olha o sinal de $\mathbf{w}\cdot\mathbf{x} + b$.
Com $\eta = 0.01$, o $\mathbf{w}_0$ ($\lVert\mathbf{w}_0\rVert = 0.013$) tem mais ou menos o tamanho de um passo,
então ele ainda influencia quais pontos saem errados no começo. Com $\eta = 1$ o primeiro passo já é muito
maior que ele. Acho que é daí que vêm a diferença de 48 para 25 erros e os 2° de diferença na direção.

**3. Começando com $\mathbf{w} = 0$ e $b = 0$, o η só muda a escala dos pesos.** Mostrei por indução. Chamo de
$(\mathbf{w}^{(t)}_1, b^{(t)}_1)$ e $(\mathbf{w}^{(t)}_2, b^{(t)}_2)$ os pesos depois de $t$ amostras nas execuções
com $\eta_1$ e $\eta_2$, e a hipótese é que $\mathbf{w}^{(t)}_2 = \tfrac{\eta_2}{\eta_1}\mathbf{w}^{(t)}_1$ e
$b^{(t)}_2 = \tfrac{\eta_2}{\eta_1} b^{(t)}_1$.

- **Base** ($t = 0$): os dois são zero, então vale para qualquer fator.
- **Passo:** para a próxima amostra $(\mathbf{x}, y)$,
  $\mathbf{w}^{(t)}_2\cdot\mathbf{x} + b^{(t)}_2 = \tfrac{\eta_2}{\eta_1}\left(\mathbf{w}^{(t)}_1\cdot\mathbf{x} + b^{(t)}_1\right)$.
  Como $\eta_2/\eta_1 > 0$, o sinal é o mesmo, então $\hat y_2 = \hat y_1$ e o erro $e = y - \hat y$ também é o
  mesmo. Daí:

$$\mathbf{w}^{(t+1)}_2 = \tfrac{\eta_2}{\eta_1}\mathbf{w}^{(t)}_1 + \eta_2\, e\,\mathbf{x}
= \tfrac{\eta_2}{\eta_1}\left(\mathbf{w}^{(t)}_1 + \eta_1\, e\,\mathbf{x}\right) = \tfrac{\eta_2}{\eta_1}\mathbf{w}^{(t+1)}_1,$$

e para o $b$ é igual. Então as duas execuções erram nos mesmos pontos, na mesma ordem, param na mesma época, e
os pesos finais só diferem pelo fator $\eta_2/\eta_1$. A fronteira é a mesma.

Também conferi isso no script com $\eta_1 = 0.01$ e $\eta_2 = 1$: deu $\mathbf{w}_1 = [0.017074,\ 0.016729]$ e
$b_1 = -0.11$, e $\mathbf{w}_2 = [1.7074,\ 1.6729]$ e $b_2 = -11$. A razão foi **exatamente 100** nas três
coordenadas, as previsões foram iguais nos 2000 pontos e as atualizações por época também (25 e depois 0).

Quando $\mathbf{w}_0 \ne 0$ a indução não funciona já na base, porque $\mathbf{w}_0 \ne \tfrac{\eta_2}{\eta_1}\mathbf{w}_0$,
e por isso no item 2 as direções ficaram um pouco diferentes. Uma coisa que eu achei interessante: a execução
com $\eta = 1$ do item 2 e a que começa do zero no item 3 deram quase o mesmo resultado
($[1.717,\ 1.665]$ e $[1.707,\ 1.673]$, as duas com 25 erros). Com um passo grande, o $\mathbf{w}_0$ pequeno
praticamente não faz diferença.

---

## Exercise 2

**Overlapping data.** Script do exercício. Ele importa a mesma `train_perceptron` do Exercise 1, sem nenhuma
mudança.

```python
--8<-- "docs/exercises/perceptron/code/ex2_overlapping.py"
```

### A — Generate the data

De novo 1000 amostras por classe, agora com $\Sigma = 1.5\,I$: a classe 0 com $\mu_0 = (3,\ 3)$ e a classe 1
com $\mu_1 = (4,\ 4)$. Agora a distância entre os centros é só $\sqrt 2 \approx 1.41$, e cada centro fica a
$0.71 / 1.22 \approx 0.58$ desvio da reta do meio. Na figura dá para ver que as duas nuvens ficam praticamente
uma em cima da outra, então **não existe reta que separe tudo**.

![Figure 4](figures/fig4.png)

### B — Train, keeping the best weights

Usei a mesma função, com $\eta = 0.01$ e um $\mathbf{w}_0 \sim \mathcal{N}(0,\ 0.01)$ novo. Dessa vez o treino
foi até as 100 épocas sem convergir: em **todas** as épocas teve entre 658 e 761 atualizações (701 em média).

| | $\mathbf{w}$ | $b$ | acurácia |
|---|---|---|---|
| Final (época 100) | [0.07538, 0.08603] | −0.41 | **63.90%** |
| Pocket (época 23) | [0.05678, 0.06367] | −0.42 | **73.30%** |

### C — Figures

Na Figura 5 os círculos vermelhos são os pontos que a fronteira **daquele painel** erra. A outra fronteira
aparece tracejada só para comparar.

![Figure 5](figures/fig5.png)

![Figure 6](figures/fig6.png)

### D — Analysis

Para entender onde cada fronteira ficou, calculei no script três coisas: quantos pontos ela classifica como
classe 1, o ângulo do $\mathbf{w}$ (o ideal seria 45°, na direção de $\mu_1 - \mu_0$) e a distância com sinal
entre o ponto médio (3.5, 3.5) e a reta (o ideal seria 0, porque a melhor reta passa por esse ponto).

| | prevê classe 1 | ângulo de $\mathbf{w}$ | distância de (3.5, 3.5) à reta | acurácia |
|---|---|---|---|---|
| Final | 81.2% | 48.8° | +1.355 | 63.90% |
| Pocket | 50.1% | 48.3° | +0.019 | 73.30% |
| Reta de Bayes ($x_1 + x_2 = 7$) | — | 45.0° | 0 | 72.60% |

**1. Por que o final e o pocket ficam tão diferentes?** O pocket chegou praticamente no máximo que uma reta
consegue nesses dados. Como as duas classes têm a mesma covariância, a melhor reta é a que passa no meio dos
dois centros, e a acurácia teórica dela é $\Phi(\lVert\mu_1 - \mu_0\rVert / 2\sigma) = \Phi(0.577) = 71.8\%$.
Nesta amostra essa reta acerta 72.6%. O pocket ficou um pouco acima (73.3%) porque ele pega a melhor reta
para esses 2000 pontos específicos. A fronteira dele passa a 0.02 do ponto médio e divide os pontos quase meio
a meio.

A fronteira final tem quase o mesmo ângulo (48.8° contra 48.3°), mas está **deslocada**: ela passa 1.36
unidades abaixo e à esquerda do ponto médio, e por isso manda 81.2% dos pontos para a classe 1. Ela acerta
quase toda a classe 1 e erra muita coisa da classe 0 (painel da esquerda da Figura 5). Então o problema dela
não é a inclinação, é a posição.

O motivo é que, com as classes sobrepostas, sempre tem ponto errado, então o perceptron **nunca para de
atualizar**. Cada erro muda o $b$ em $\pm\eta$ e o $\mathbf{w}$ em $\pm\eta\mathbf{x}$: um ponto da classe 1
errado puxa a reta para baixo e para a esquerda, e um ponto da classe 0 errado puxa para cima e para a
direita. Na região onde as nuvens se misturam isso não acaba nunca, e a reta fica indo e voltando em volta da
posição boa. A fronteira da época 100 é simplesmente onde a reta estava quando o treino acabou, e nada garante
que seja uma posição boa. É para isso que serve o pocket: ele não muda o treino, só guarda o melhor momento.

**2. Comparando as Figuras 3 e 6.** Na Figura 3 a acurácia chega a 100% e o treino para sozinho porque uma
época inteira passa sem erro. Na Figura 6 a acurácia corrente fica oscilando entre 59.8% e 71.4% do começo ao
fim, sem subir, enquanto a do pocket já estava em 73.1% no fim da primeira época e chegou a 73.3% na época 23,
e não mudou mais.

O teorema de convergência diz que, **se existir** um $(\mathbf{w}^*, b^*)$ com
$y_i'(\mathbf{w}^*\cdot\mathbf{x}_i + b^*) \ge \gamma > 0$ para todo $i$ (usando $y' = \pm 1$), o perceptron faz
no máximo $(R/\gamma)^2$ atualizações, ou seja, ele para depois de um número finito de passos e com erro zero.
A hipótese que falha aqui é a de que os dados são **linearmente separáveis** (margem $\gamma > 0$). Como
nenhuma reta acerta todos os pontos, o teorema não garante nada, e o que acontece na prática é que os pesos
ficam mudando para sempre. E mesmo quando os dados não são separáveis, o teorema não fala nada sobre achar a
reta que erra menos.

**3. Mais épocas ou um η menor resolveriam?** Não. Para testar, rodei de novo com **500 épocas**, o mesmo
$\mathbf{w}_0$ e três valores de η:

| $\eta$ | acurácia final | corrente nas últimas 100 épocas (mín–máx) | atualizações/época | pocket |
|---|---|---|---|---|
| 1.0 | 63.35% | 59.45% – 71.40% | 700 | 73.40% |
| 0.01 | 66.50% | 59.45% – 71.35% | 700 | 73.35% |
| 0.001 | 63.45% | 59.75% – 66.80% | 701 | 73.35% |

- **Mais épocas:** o treino só para quando passa uma época sem erro, e isso nunca vai acontecer, porque sempre
  tem ponto que nenhuma reta acerta. Rodar mais épocas é só repetir o mesmo vai e volta. Com 500 épocas a
  acurácia corrente ficou oscilando na mesma faixa que com 100.
- **η menor:** pelo item 3 do Exercise 1, quando o $\mathbf{w}_0$ é bem pequeno o η basicamente só muda a
  escala de $(\mathbf{w}, b)$, e a fronteira só depende da direção de $\mathbf{w}$ e de $b/\lVert\mathbf{w}\rVert$.
  Então diminuir o η não deixa os passos menores em relação aos pesos, porque os pesos também ficam menores. Os
  erros acontecem quase na mesma sequência, e é por isso que as três linhas da tabela ficaram tão parecidas. O
  que poderia fazer os pesos pararem de oscilar seria um η que diminui ao longo do treino (tipo
  $\eta_t \propto 1/t$), ou usar uma função de perda contínua (como a logística ou a hinge) em vez de olhar só
  o sinal. Mas, com o perceptron do jeito que ele é, a solução é o pocket.

---

## Results summary

| # | Quantity | Value |
| --- | --- | --- |
| 1 | Exercise 1 — final $\mathbf{w}$ and $b$ | $\mathbf{w}$ = [0.03189, 0.02874], $b$ = −0.20000 |
| 2 | Exercise 1 — epochs to convergence | 2 (48 atualizações na 1ª época, 0 na 2ª) |
| 3 | Exercise 1 — final accuracy | 1.0000 (100%) |
| 4 | Exercise 1 — epochs and final accuracy with $\eta = 1.0$ | 2 épocas, 1.0000 (100%) |
| 5 | Exercise 2 — final $\mathbf{w}$ and $b$ | $\mathbf{w}$ = [0.07538, 0.08603], $b$ = −0.41000 |
| 6 | Exercise 2 — accuracy of the final weights | 0.6390 |
| 7 | Exercise 2 — accuracy of the pocket weights | 0.7330 |
| 8 | Exercise 2 — epoch at which the pocket best occurred | 23 |
