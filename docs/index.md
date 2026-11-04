# Redes Neurais e Deep Learning

!!! info inline end "Edição"

    **2026.2** · Insper

    Lucas Sato

Portfólio dos exercícios da disciplina de **Artificial Neural Networks and Deep
Learning**. Cada entrega é um relatório próprio — com o código que a gerou, as figuras
e os números que sustentam cada afirmação.

O fio condutor não é a ferramenta, é a pergunta que vem antes dela: como os dados estão
distribuídos, o que a geometria deles já revela sobre a dificuldade do problema, e o que
precisa acontecer com eles antes de uma rede conseguir aprender qualquer coisa.

## Exercícios

<div class="grid cards" markdown>

-   :material-chart-scatter-plot:{ .lg .middle } &nbsp; **1. Data**

    ---

    Geometria e espalhamento de nuvens de pontos, não-linearidade em 5 dimensões e
    pré-processamento do Spaceship Titanic para uma rede com ativação `tanh`.

    *Três scripts · seis figuras · 13 métricas*

    [:octicons-arrow-right-24: Ler o relatório](exercises/data/index.md)

-   :material-vector-point:{ .lg .middle } &nbsp; **2. Perceptron**

    ---

    Perceptron implementado do zero: converge quando os dados são separáveis e
    fica oscilando quando se sobrepõem, que é onde entra o *pocket*.

    *Perceptron em NumPy · seis figuras · 8 métricas*

    [:octicons-arrow-right-24: Ler o relatório](exercises/perceptron/index.md)

-   :material-graph-outline:{ .lg .middle } &nbsp; **3. MLP**

    ---

    MLP implementado do zero: backpropagation à mão, um problema que uma reta não
    resolve, três classes com softmax e o teste de uma segunda camada oculta.

    *MLP em NumPy · seis figuras · 9 métricas*

    [:octicons-arrow-right-24: Ler o relatório](exercises/mlp/index.md)

-   :material-blur:{ .lg .middle } &nbsp; **4. VAE**

    ---

    VAE no MNIST com os dois termos da perda medidos separadamente e um espaço latente
    de duas dimensões, desenhado para medir o que se perde ao comprimir cada imagem.

    *VAE em PyTorch · seis figuras · 4 métricas*

    [:octicons-arrow-right-24: Ler o relatório](exercises/vae/index.md)

</div>

## Andamento

- [x] **Data** — entregue em 10/set/2026
- [x] **Perceptron** — entregue em 22/set/2026
- [x] **MLP** — entregue em 13/out/2026
- [x] **VAE** — entregue em 03/nov/2026

## Como reproduzir

Todo relatório é reproduzível a partir de um checkout limpo. Os scripts fixam
`rng = np.random.default_rng(42)` (no VAE, `torch.manual_seed(42)`) e leem os dados de
caminhos relativos ao próprio arquivo, então rodam de qualquer lugar. O MNIST é baixado
pelo próprio script do VAE na primeira execução.

=== "Rodar os scripts"

    ``` { .bash .copy }
    git clone https://github.com/lucassatoco/DLRN_2026.2.git
    cd DLRN_2026.2
    python -m venv .venv && .venv/Scripts/activate
    pip install -r requirements.txt
    cd docs/exercises/data/code
    python ex1_point_clouds.py
    python ex2_nonlinearity.py
    python ex3_preprocessing.py
    cd ../../perceptron/code
    python ex1_separable.py
    python ex2_overlapping.py
    cd ../../mlp/code
    python ex2_binary.py
    python ex3_multiclass.py
    python ex4_depth.py
    cd ../../vae/code
    python ex1_mnist.py
    python ex2_latent2d.py
    ```

=== "Servir o site localmente"

    ``` { .bash .copy }
    pip install -r requirements.txt
    mkdocs serve
    ```

!!! note "Uso de IA"

    O uso de IA está declarado no front matter de cada relatório, no campo `ai_use`,
    como a disciplina exige. Colaboração é permitida; não declarar é o que gera
    problema.

## Ferramentas

`numpy` · `pandas` · `matplotlib` · `scikit-learn` — esta última apenas para PCA e
pré-processamento no exercício de Data e, no MLP, para gerar os dados, dividir treino e
teste, a matriz de confusão e as baselines de regressão logística. O perceptron e o MLP
são escritos do zero em NumPy, sem nenhum modelo pronto. No VAE uso `torch` e `torchvision`
(o enunciado libera o autograd), mas a reparametrização, os dois termos da perda e a
amostragem são escritos à mão; `scipy` entra só para a inversa da CDF normal.

Site construído com [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/){:target='_blank'}
e publicado via GitHub Pages.

---

[:octicons-mark-github-16: Repositório](https://github.com/lucassatoco/DLRN_2026.2){:target='_blank'} ·
[:octicons-book-16: Enunciados da disciplina](https://insper.github.io/ann-dl/2026.2/){:target='_blank'}
