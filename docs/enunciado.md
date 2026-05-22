# Carro Autônomo

Neste exercício-programa, o agente é um carrinho que precisa aprender a **pilotar uma pista 2D** usando **aprendizado por reforço tabular**. Você implementará o **Q-Learning** e o analisará em pistas de dificuldade crescente, observando como o agente aprende a coordenar velocidade e direção a partir apenas de sensores tipo LIDAR.

A continuidade com o EP anterior (busca informada com A*) é proposital: lá, o agente conhecia o ambiente e planejava a rota; aqui, o ambiente é desconhecido e o agente precisa aprender por interação. Mesmo domínio (grid 2D), formato similar de I/O, mas paradigma fundamentalmente diferente.

Utilizar este código-fonte como base: https://github.com/senac-ia/rf-carro-autonomo

## 1. O Ambiente

### 1.1 Pistas

Uma **pista** é um grid 2D binário com os seguintes elementos:

- **Parede (🧱):** zona intransponível.
- **Asfalto (⚪️):** zona pilotável.
- **Largada (🟢):** posição inicial do carro.
- **Linha de chegada (🏁):** alvo.

Exemplo de pista (formato `entrada.txt`):

```
🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱
🧱 🟢 ⚪️ ⚪️ ⚪️ ⚪️ ⚪️ ⚪️ ⚪️ ⚪️ 🏁 🧱
🧱 ⚪️ ⚪️ ⚪️ ⚪️ ⚪️ ⚪️ ⚪️ ⚪️ ⚪️ ⚪️ 🧱
🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱
```

> O EP fornece um conjunto de **18 pistas** (`pista_01.txt` a `pista_18.txt`) organizadas em três níveis de dificuldade (ver `descricao_pistas.md` para o design detalhado de cada uma):
- **01–04 (fáceis):** progressão pedagógica — cada pista introduz uma habilidade nova que o Q-Learning precisa aprender (reagir a parede frontal, generalizar curvas, ajuste fino de ângulo, U-turn com chicane). Corredor 3–4 células. Boas para depurar e para o baseline da T4.1.
- **05–12 (médias):** combinam vários elementos (chicanes, curvas em sequência, mudanças de direção). Corredor 3–4 células. A `pista_07.txt` é usada na T4.2.
- **13–18 (difíceis):** corredor pode chegar a 2 células, com várias mudanças de direção. Para experimentos opcionais ou para investigar limites do tabular.
> 
> 
> Você pode também criar pistas adicionais para exploração.
> 

### 1.2 Carro

O carro tem o seguinte estado físico:

- **Posição** $(x, y) \in \mathbb{R}^2$ (contínua, mesmo em grid discreto).
- **Ângulo** $\theta \in [0, 2\pi)$.
- **Velocidade** $v \in [0, V_{\max}]$.

A cada passo, a posição é atualizada por: $x \leftarrow x + v \cos\theta$, $y \leftarrow y + v \sin\theta$.

A célula atual do grid é dada por arredondamento de $(x, y)$. Se essa célula é parede, considera-se **colisão** (recompensa fortemente negativa, episódio termina).

> Veja [`docs/anexo_c_velocidade.md`](anexo_c_velocidade.md) para uma discussão detalhada sobre velocidade — é o componente mais sutil do problema.
> 

### 1.3 Ações

Espaço discreto de **5 ações**:

| Ação | Efeito |
| --- | --- |
| 0 | Nada (mantém velocidade e ângulo) |
| 1 | Acelerar ($v \leftarrow \min(v + 0{,}5,\ V_{\max})$) |
| 2 | Frear ($v \leftarrow \max(v - 0{,}5,\ 0)$) |
| 3 | Virar à esquerda ($\theta \leftarrow \theta - 30°$) |
| 4 | Virar à direita ($\theta \leftarrow \theta + 30°$) |

Sugestão: $V_{\max} = 2{,}0$ (em células por passo).

## 2. Representação do Estado e Discretização

A representação observável é um vetor baixo-dimensional baseado em **sensores tipo LIDAR** (ver [`docs/anexo_a_lidar.md`](anexo_a_lidar.md)):

```
estado = [d_0, d_+30, d_-30, d_+60, d_-60, v_norm]
```

onde:

- $d_\alpha$ é a distância (em células, normalizada por algum fator) até a parede mais próxima na direção $\theta + \alpha$. Use 5 raios (frente, ±30°, ±60°).
- $v_\text{norm} = v / V_{\max}$.

Ou seja: **estado é um vetor de 6 floats em $[0, 1]$**.

### 2.1 Discretização

Como o Q-Learning tabular precisa de uma **tabela $Q[s, a]$** indexada por estados discretos, você precisa **converter o vetor de 6 floats em uma chave discreta**.

**Estratégia adotada neste EP:** divida cada componente em $K = 5$ baldes de tamanho igual em $[0, 1]$:

```python
def discretize(obs, n_bins=5):
    # obs ∈ [0, 1]^6
    return tuple(min(int(v * n_bins), n_bins - 1) for v in obs)
```

O resultado é uma tupla de 6 inteiros em $\{0, 1, 2, 3, 4\}$, que serve como chave da tabela $Q$ (use um `dict[chave, np.ndarray(5)]` ou um `np.ndarray` 7-dimensional).

**Por que $K = 5$ é uma boa escolha para este problema?**

1. **Casa com a granularidade da velocidade.** O carro tem $V_{\max} = 2{,}0$ e incrementos de $0{,}5$, então $v$ assume apenas 5 valores distintos: $\{0;\ 0{,}5;\ 1{,}0;\ 1{,}5;\ 2{,}0\}$. Normalizada, vira $\{0;\ 0{,}25;\ 0{,}5;\ 0{,}75;\ 1{,}0\}$ — exatamente 5 baldes, sem agregar nem fragmentar valores físicos.
2. **Resolução suficiente para os sensores LIDAR.** Com $K = 5$, cada balde cobre 20% do alcance máximo (2 células de 10). É grossa o bastante para o agente aprender em poucos episódios, e fina o bastante para distinguir “colado na parede” (balde 0) de “com folga” (baldes 1+).
3. **Tamanho de tabela manejável.** Com 6 dimensões, há até $5^6 = 15{.}625$ estados; com 5 ações, são $78{.}125$ entradas na tabela $Q$. Treinamento de 30.000 episódios popula uma fração disso e converge rapidamente.

> Discretizações mais finas (ex.: $K = 8$ ou $K = 10$) explodem o número de estados ($8^6 \approx 262$ mil; $10^6 = 1$ milhão) e tornam o aprendizado muito mais lento sem ganho prático aqui, porque a velocidade só tem 5 níveis e o LIDAR já é amostrado em passos de $0{,}1$ célula no *ray casting*. Discretizações mais grosseiras ($K = 3$) agregam demais — o agente não consegue separar “perto da parede” de “colado na parede” e colide com frequência.
> 
> Por essas razões, neste EP **$K = 5$ é fixo** e o foco do trabalho está no **Q-Learning** e no seu comportamento em pistas de dificuldade crescente (Tarefas 4.1 e 4.2).
> 

## 3. Recompensas

Recompensa esparsa não funciona aqui. Use a seguinte estrutura:

1. **Avanço de progresso:** a cada passo, $r_\text{progresso} = +\Delta s$, onde $\Delta s$ é a variação de distância percorrida ao longo do caminho da pista (calculada por BFS desde a largada). Pode ser positivo (avançou) ou zero (não progrediu).
2. **Custo de tempo:** $r_\text{tempo} = -0{,}1$ por passo (incentivo a terminar rápido).
3. **Colisão com parede:** $r_\text{colisao} = -100$ e **episódio termina**.
4. **Cruzou a linha de chegada 🏁:** $r_\text{chegada} = +500$ e episódio termina.
5. **Limite de passos do episódio (ex.: 500):** episódio termina sem bônus.

Recompensa total por passo: $r = r_\text{progresso} + r_\text{tempo}$ (mais um dos terminais quando aplicável).

> O reward shaping baseado em progresso já vem implementado no starter code. Veja seção 4.3 do relatório se quiser experimentar com recompensa esparsa.
> 

## 4. Tarefas

### 4.1 Q-Learning Baseline

Implemente Q-Learning com $\varepsilon$-greedy. Treine na pista `pista_03.txt` (curva moderada).

> Veja **`docs/qlearning.md`** para a matemática do algoritmo (atualização TD, $\varepsilon$-greedy, por que é off-policy), pseudocódigo e dicas de implementação em Python.

Hiperparâmetros sugeridos como ponto de partida:

- **Episódios de treinamento:** 30.000
- **Limite de passos por episódio:** 500
- **Discretização:** $K = 5$ baldes por dimensão
- **Taxa de aprendizado** $\alpha$: 0,1
- **Desconto** $\gamma$: 0,99
- **Exploração** $\varepsilon$: decai linearmente de 1,0 a 0,05 nos primeiros 80% dos episódios

Ao final, **avalie a política aprendida** com $\varepsilon = 0$ (gulosa) e gere `q_learning.txt` com:

```
=== Pista: pista_03.txt ===
Algoritmo: Q-Learning
Episódios de treinamento: 30000
Discretização: K=5
Tempo de chegada (passos): 27
Velocidade média: 1.42
Velocidade máxima atingida: 2.0
Recompensa total: 478.4
Sucesso: SIM
```

### 4.2 Q-Learning em pista com risco (Cliff-style)

Essa tarefa é o coração pedagógico do EP — investiga como o Q-Learning se comporta em uma pista onde **errar custa caro**, evocando a essência do experimento *Cliff Walking* do Sutton & Barto.

Use a pista `pista_07.txt` (curva apertada — alto risco de colisão durante exploração).

Treine o Q-Learning com a **mesma configuração** da T4.1 ($\alpha=0{,}1$, $\gamma=0{,}99$, $\varepsilon$ decaindo de 1,0 a 0,05 em 80% de 30.000 episódios, $K=5$).

Analise e reporte:

- **Histórico de aprendizado** (recompensa média por episódio em janela móvel de 100, salvo no pickle e reportado no relatório como tabela com marcos ou ASCII). Compare com o histórico da T4.1 — a curva é mais ruidosa? Demora mais para estabilizar?
- **Recompensa média durante o treinamento** (com $\varepsilon$-greedy ativo) vs. **recompensa final em avaliação gulosa** ($\varepsilon = 0$). A diferença é maior do que na T4.1? Por quê?
- **Velocidade média** da política aprendida. O agente fica mais conservador (devagar) ou mais agressivo do que na T4.1?
- **Trajetória visual** — use a animação no terminal (`renderizar_episodio` em `src/visualize.py`) para inspecionar a política final. O agente passa colado nas paredes ou mantém folga? Descreva o que observou (capturar o terminal em texto ou descrever em prosa basta).

Gere `cliff.txt` com o resumo da política treinada. Discuta no relatório como o **trade-off entre exploração e explotação** muda quando colidir tem custo alto, e como isso aparece no comportamento aprendido (velocidade, distância das paredes, taxa de colisão durante o treinamento).


## 5. Saída do Programa

O programa deve gerar, para cada experimento, um arquivo de saída listando o desempenho final:

- **`q_learning.txt`:** resultado da T4.1 (Q-Learning baseline) em `pista_03.txt`.
- **`cliff.txt`:** resultado da T4.2 (Q-Learning em pista de risco) em `pista_07.txt`.

Formato sugerido para cada arquivo:

```
=== Pista: pista_03.txt ===
Algoritmo: Q-Learning
Episódios de treinamento: 30000
Discretização: K=5
Estados populados: 1247
Tempo de chegada (passos): 27
Velocidade média: 1.42
Recompensa total: 478.4
Sucesso: SIM
```

## 6. Relatório (README.md)

### 1. Modelagem do MDP e Q-Learning Baseline (T4.1)

- **Espaço de estados (após discretização $K = 5$):** quantos estados, em teoria? E na prática (após o treinamento)?
- **Espaço de ações:** justifique se 5 ações são suficientes.
- **Função de recompensa:** explique como você implementou o reward shaping.
- **Como você está armazenando $Q[s,a]$ internamente** (dicionário, array NumPy multidimensional)?
- **Resultado do baseline:** quantos passos o Q-Learning leva para completar a pista? Velocidade média atingida? Perfil de uso de cada ação?

### 2. Q-Learning em pista de risco — Cliff-style (T4.2)

- Como o desempenho do Q-Learning muda em uma pista com alto risco de colisão? Compare curva de aprendizado, recompensa final e velocidade média com os resultados da T4.1.
- A diferença entre a recompensa durante o treinamento (com $\varepsilon$-greedy) e a recompensa em avaliação gulosa ($\varepsilon = 0$) é maior aqui? Como isso reflete o efeito de **explorar perto de paredes**?
- Discuta com base nas trajetórias observadas via animação no terminal (`renderizar_episodio` em `src/visualize.py`) — o agente passa colado nas paredes ou mantém folga? Como isso se relaciona com o comportamento off-policy do Q-Learning (que aprende a política gulosa enquanto explora aleatoriamente)?

---

## 7. Restrições de Implementação

- **Linguagem:** Python 3.10+.
- **Bibliotecas permitidas:** `numpy`, `tqdm`. A visualização do agente é via animação no terminal (`src/visualize.py`), sem dependências de imagem.
- **Bibliotecas proibidas:** `gymnasium`, `stable-baselines3`, `ray[rllib]`, `tianshou`, `torch` (não precisa para tabular), ou qualquer biblioteca de RL pronta. **Você deve implementar o Q-Learning do zero**, incluindo a função de discretização.
- O ambiente do carro vem fornecido no starter code (`src/env.py`). Você não precisa reimplementá-lo.

---

## A Entrega

### Código fonte

Deverá ser entregue o **repositório no GitHub**. Deverá ser implementado usando o ambiente de testes em https://github.com/senac-ia/rf-carro-autonomo

Não é obrigatório, mas preferencialmente usar Python. A correção do professor considera que o algoritmo deve rodar em sua máquina local, portanto, deverá ter as instruções de como rodar e dependências no `README.md`.

**Não será permitido o uso de bibliotecas de software de Inteligência Artificial pronta** (`gymnasium`, `stable-baselines3`, `ray[rllib]`, `tianshou`, etc.). Você deve implementar o Q-Learning **do zero**. Você pode usar `numpy` e `tqdm` (já listados em `requirements.txt`).

Pode ser baseado no starter code fornecido em https://github.com/senac-ia/rf-carro-autonomo que contém:

- Parser de pistas em emojis (`src/track.py`)
- Ambiente do carro com física e LIDAR (`src/env.py`)
- Visualização animada (`src/visualize.py`)
- 18 pistas (`pistas/pista_01.txt` a `pistas/pista_18.txt`), de retas simples a circuitos com múltiplas mudanças de direção
- Esqueleto de `solucao.py` para preencher

### Salvamento de modelos treinados

Como os tempos de treinamento podem ser longos (30.000 episódios em pistas mais complexas leva vários minutos), você **deve salvar os modelos treinados em disco**, no diretório `/treinamento` na raiz do projeto. Use **`pickle`** da biblioteca padrão do Python (ver [`docs/anexo_b_pickle.md`](anexo_b_pickle.md)).

Estrutura esperada na raiz do projeto:

```
seu-projeto/
├── README.md
├── solucao.py
├── src/
├── pistas/
└── treinamento/
    ├── q_learning_K5_pista_03.pkl  ← baseline Q-Learning (T4.1)
    └── q_learning_K5_pista_07.pkl  ← Q-Learning em pista de risco (T4.2)
```

Esses arquivos devem ser **commitados no repositório**. Isso permite ao professor reproduzir as avaliações sem re-treinar.

### O que será avaliado

### Explicar a modelagem em apresentação

- **Representação do espaço de estados:**
    - Como você implementou a discretização do vetor de 6 floats com $K = 5$?
    - Qual o tamanho real da tabela $Q$ ao final do treinamento? Como esse número se compara ao máximo teórico de $5^6 \times 5 = 78{.}125$ entradas?
- **Espaço de ações:** como as 5 ações foram codificadas?
- **Função de recompensa:** como o reward shaping foi implementado? Você experimentou variações?
- **Política de exploração:** schedule de $\varepsilon$, justificativa.

> **Atenção:** fazer cópia do algoritmo apenas e explicar o que é o conceito **não vale**. O trabalho requer a explicação de como o conceito foi **modelado e implementado para este problema específico** (pilotar um carrinho).
> 

### Em código

- **Implementação do Q-Learning** do zero.
- **Função de discretização** com $K = 5$ (fornecida como ponto de partida no enunciado).
- **Loop de treinamento** que registra histórico de recompensas por episódio (salvo no pickle do modelo).
- **Loop de avaliação** com $\varepsilon = 0$ que gera os arquivos de saída descritos na seção 5.
- **Inspeção da política final** via animação no terminal (`renderizar_episodio` em `src/visualize.py`) — descreva no relatório o que observou para a política treinada em cada pista.
- **Salvamento e carregamento** dos modelos via pickle no diretório `/treinamento`.

### Critérios de avaliação

- Explicação da lógica do problema e da modelagem do MDP.
- Explicação da discretização adotada (por que $K = 5$ funciona bem aqui).
- Explicação das funções principais e estrutura do código.
- Demonstração dos resultados (histórico de aprendizado em formato textual, animação do agente no terminal, tabelas de avaliação).
- **Análise crítica** — especialmente do comportamento do Q-Learning na pista Cliff-style (efeito do risco sobre exploração e política aprendida).
- Criatividade — extensões além do mínimo, exploração de variações na função de recompensa ou em outras pistas.

### Política de uso de ferramentas

Este trabalho deve seguir:

[Política de uso de ferramentas generativas de IA](https://www.notion.so/...)

[Política antiplágio](https://www.notion.so/...)

---

## Anexos

Conteúdo de apoio em arquivos separados:

- [`docs/anexo_a_lidar.md`](anexo_a_lidar.md) — o que são sensores tipo LIDAR (real e simulado).
- [`docs/anexo_b_pickle.md`](anexo_b_pickle.md) — como salvar e carregar modelos treinados com `pickle`.
- [`docs/anexo_c_velocidade.md`](anexo_c_velocidade.md) — discussão detalhada sobre a velocidade do carro.
