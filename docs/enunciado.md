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

> Veja o **Anexo C** para uma discussão detalhada sobre velocidade — é o componente mais sutil do problema.
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

A representação observável é um vetor baixo-dimensional baseado em **sensores tipo LIDAR** (ver Anexo A):

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

Como os tempos de treinamento podem ser longos (30.000 episódios em pistas mais complexas leva vários minutos), você **deve salvar os modelos treinados em disco**, no diretório `/treinamento` na raiz do projeto. Use **`pickle`** da biblioteca padrão do Python (ver Anexo B).

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

## Anexo A: O que são sensores tipo LIDAR

A seção 2 do enunciado se refere a “sensores tipo LIDAR” como representação do estado do carro. Este anexo explica o conceito para quem nunca encontrou o termo.

### A.1 LIDAR no mundo real

**LIDAR** é a sigla para **Li**ght **D**etection **a**nd **R**anging — detecção e medição por luz. É um sensor que mede **distâncias** disparando feixes de laser e medindo o tempo que a luz leva para bater num objeto e voltar. A ideia é a mesma do **sonar** (usa som) ou do **radar** (usa ondas de rádio), só que com luz: dispara → reflete → recebe → calcula distância. Como a velocidade da luz é constante e conhecida, **tempo de voo × velocidade da luz / 2** dá a distância até o obstáculo.

**Onde aparece:**

- **Carros autônomos** (Waymo, Cruise): aquele “domo” giratório no teto do carro é um LIDAR.
- **Robôs aspiradores** (Roomba, Roborock): mapeiam sua casa.
- **Drones autônomos**: para evitar obstáculos.
- **iPhones recentes**: têm um mini-LIDAR para realidade aumentada e foco em fotos.
- **Topografia e arqueologia**: aviões com LIDAR mapeiam relevo abaixo de florestas densas.

A saída típica de um LIDAR é um **vetor de distâncias**, uma para cada direção em que ele aponta:

```
Direção        Distância
0°  (frente)   8,2 m
+30°           5,1 m
-30°           12,5 m
+60°           2,4 m
-60°           ∞ (não bateu em nada dentro do alcance)
```

### A.2 LIDAR simulado no EP

No ambiente do EP, o “LIDAR” é **simulado** — não existe luz nem laser de verdade. O que `src/env.py` faz é **ray casting**:

1. A partir da posição do carro $(x, y)$ e seu ângulo $\theta$, emitimos **5 raios** virtuais nas direções $\theta + 0°$, $\theta \pm 30°$, $\theta \pm 60°$.
2. Para cada raio, andamos passo a passo em pequenos incrementos (`step = 0,1` célula), checando se a célula atual é parede.
3. Quando bate numa parede ou ultrapassa o alcance máximo (10 células), registramos a distância percorrida.
4. O **estado do agente** vira um vetor de 6 floats:

```
[d_frente, d_+30°, d_-30°, d_+60°, d_-60°, velocidade_normalizada]
```

Esse é o **único input que o agente vê**. Ele não vê o mapa, não sabe onde está, não sabe onde é a chegada — só sabe “o que tem perto na minha frente e nos meus lados”.

> 💡 É exatamente isso que um carro real “vê” pelo LIDAR físico. A diferença é que aqui simulamos via *ray casting* num grid 2D em vez de usar luz física.
> 

### A.3 Limitações do LIDAR (real e simulado)

Vale conhecer os limites:

- **Vidro e materiais transparentes:** LIDAR real tem dificuldade — o laser atravessa o vidro.
- **Chuva forte, neve, neblina:** as gotículas refletem o laser e geram leituras falsas.
- **Apenas um plano:** um LIDAR 2D só vê uma “fatia” horizontal. Em carros reais isso exige LIDAR 3D ou múltiplos sensores em alturas diferentes.
- **Custo:** um LIDAR automotivo decente custa milhares de dólares. É parte do motivo de a Tesla ter apostado em câmeras + visão computacional em vez de LIDAR.

### A.4 Recursos para aprofundar

- *Wikipedia: LIDAR* — boa visão geral histórica e técnica.
- *Velodyne, Ouster, Luminar* — fabricantes; sites têm whitepapers acessíveis.
- *Self-Driving Cars Specialization* (Coursera, Univ. of Toronto) — curso aborda integração LIDAR + percepção em detalhe.

---

## Anexo B: Salvando modelos treinados com `pickle`

Como o EP exige salvar os modelos treinados no diretório `/treinamento`, este anexo explica como fazer isso usando `pickle` — uma biblioteca da **biblioteca padrão do Python** (não precisa instalar nada).

### B.1 O que é pickle

`pickle` é o módulo do Python que **serializa** objetos: transforma uma estrutura de dados Python (dicionário, lista, classe, array NumPy) em uma sequência de bytes que pode ser salva em arquivo e, depois, **desserializada** de volta no estado original.

Em uma frase: pickle congela um objeto Python em disco, e descongela quando você quiser.

> 💡 **Analogia:** é como tirar uma “foto” do seu modelo treinado e poder revivê-la depois sem precisar re-treinar tudo.
> 

### B.2 Uso básico

**Salvar um objeto:**

```python
import pickle

dados = {"q_table": minha_q_table, "config": {"alpha": 0.1, "gamma": 0.99}}

with open("treinamento/q_learning.pkl", "wb") as f:   # "wb" = write binary
    pickle.dump(dados, f)
```

**Carregar um objeto:**

```python
import pickle

with open("treinamento/q_learning.pkl", "rb") as f:   # "rb" = read binary
    dados = pickle.load(f)

minha_q_table = dados["q_table"]
config = dados["config"]
```

Pronto. Não há mais nada de fundamental.

### B.3 O que salvar para o Q-Learning

Salve um dicionário com tudo que você precisa para reproduzir o agente:

```python
estado_para_salvar = {
    "q_table": agent.q_table,           # dict {chave_discreta: array} ou np.ndarray
    "discretization_K": 5,              # configuração da discretização
    "n_episodes_trained": 30_000,
    "rewards_history": rewards,         # lista de recompensas por episódio
    "config": {"alpha": 0.1, "gamma": 0.99, "eps_start": 1.0, "eps_end": 0.05},
    "seed": 42,
    "track_used": "pistas/pista_03.txt",
}
with open("treinamento/q_learning_K5_pista_03.pkl", "wb") as f:
    pickle.dump(estado_para_salvar, f)
```

### B.4 Lógica recomendada para `solucao.py`

Implemente a seguinte lógica para evitar re-treinar a cada execução:

```python
import os
import pickle
from pathlib import Path

TREINAMENTO_DIR = Path("treinamento")
TREINAMENTO_DIR.mkdir(exist_ok=True)

def treinar_ou_carregar(nome, treinar_fn, recarregar=False):
    """
    Se 'treinamento/{nome}.pkl' existe e recarregar=False, carrega.
    Caso contrário, chama treinar_fn() e salva o resultado.
    """
    arquivo = TREINAMENTO_DIR / f"{nome}.pkl"
    if arquivo.exists() and not recarregar:
        print(f"Carregando{arquivo} ...")
        with open(arquivo, "rb") as f:
            return pickle.load(f)
    else:
        print(f"Treinando{nome} ...")
        resultado = treinar_fn()
        with open(arquivo, "wb") as f:
            pickle.dump(resultado, f)
        print(f"Salvo em{arquivo}")
        return resultado

# Uso:
q_baseline = treinar_ou_carregar("q_learning_K5_pista_03", lambda: treinar_q_learning(env, K=5))
q_cliff = treinar_ou_carregar("q_learning_K5_pista_07", lambda: treinar_q_learning(env_07, K=5))
```

Para forçar re-treinamento (útil ao depurar), passe `recarregar=True` ou simplesmente delete o arquivo `.pkl`.

### B.5 Cuidados com pickle

- **Nunca abra um pickle de fonte desconhecida.** O processo de unpickling pode executar código arbitrário — é um vetor clássico de ataque. Para os modelos que você mesmo gerou, sem problema.
- **Compatibilidade entre versões do Python:** pickles em Python 3.10+ são geralmente intercompatíveis, mas pickles entre Python 2 e 3 quebram. Para este EP isso não é problema (use Python 3.10+).
- **Tamanho dos arquivos:** tabelas $Q$ tabulares costumam ficar entre dezenas e centenas de KB — sem problema para commitar no GitHub.
- **Reprodutibilidade:** salve **junto com o modelo** os hiperparâmetros e a seed usada.

### B.6 Documentação oficial

- [docs.python.org/3/library/pickle.html](https://docs.python.org/3/library/pickle.html) — referência completa.

---

## Anexo C: A Velocidade do Carro

Velocidade ($v$) parece um conceito óbvio, mas no contexto deste EP tem sutilezas que vale entender bem — porque é justamente o componente do estado que torna o problema interessante (e difícil).

### C.1 Definição operacional

A velocidade $v$ é um **escalar não-negativo** que mede quanto o carro avança por passo de simulação, na direção em que ele está apontando. A cada chamada de `env.step(action)`, o carro se move:

$$
x_{novo} = x + v \cdot \cos\theta
$$

$$
y_{novo} = y + v \cdot \sin\theta
$$

A direção do movimento vem do ângulo $\theta$. A velocidade é só a **magnitude** do deslocamento.

> 💡 Em física tradicional, velocidade é um vetor (magnitude + direção). Aqui, separamos as duas componentes: $v$ guarda a magnitude, $\theta$ guarda a direção. Isso simplifica a física e o controle.
> 

### C.2 Unidade e limites

A unidade é **células do grid por passo de simulação**.

- $v = 1{,}0$ → o carro atravessa **uma célula inteira a cada passo**.
- $v = 0{,}5$ → meia célula por passo.
- $v = 0$ → parado.
- $v = V_{\max} = 2{,}0$ → duas células por passo (limite máximo).

Não há unidade de tempo “real” no problema — um “passo” é uma unidade abstrata.

### C.3 Como o agente controla a velocidade

A velocidade muda **apenas por ação do agente**. Não há fricção, não há inércia continuada — se o agente não fizer nada, $v$ permanece igual.

| Ação | Efeito em $v$ |
| --- | --- |
| 0 (nada) | $v$ inalterada |
| 1 (acelerar) | $v \leftarrow \min(v + 0{,}5,\ V_{\max})$ |
| 2 (frear) | $v \leftarrow \max(v - 0{,}5,\ 0)$ |
| 3 (virar esquerda) | $v$ inalterada (só $\theta$ muda) |
| 4 (virar direita) | $v$ inalterada (só $\theta$ muda) |

Isso é uma **física idealizada** — em um carro real, frenagem leva tempo, há fricção do ar, há inércia. No EP, ignoramos tudo isso para simplificar o aprendizado.

### C.4 Sem marcha-ré

Note: $v \in [0,\ V_{\max}]$. **Nunca negativa**. O carro só vai para frente; para mudar de direção, precisa virar. Isso modela um carro de F1, não um carro de rua.

### C.5 Velocidade no estado observável

A velocidade entra na **observação do agente** como o sexto componente do vetor de estado:

```
estado = [d_0, d_+30, d_-30, d_+60, d_-60, v_norm]
```

onde $v_{norm} = v / V_{\max}$ — normalizada para $[0, 1]$, igual aos sensores LIDAR. Isso é importante: tabelas com discretização funcionam melhor quando todas as features estão na mesma escala.

> 💡 **Por que o agente precisa saber a própria velocidade?** Porque a melhor ação depende dela. Andando devagar perto de uma curva, é seguro continuar acelerando. Andando rápido, é prudente frear antes. Sem ver a velocidade, o agente não conseguiria distinguir essas situações.
> 

### C.6 Por que velocidade é o componente sutil do problema

Os 5 sensores LIDAR são “fáceis de entender” — distâncias até paredes. A velocidade é mais traiçoeira por três motivos:

### 1. Tem efeito acumulativo

Acelerar uma vez muda $v$ em apenas $+0{,}5$ — efeito imediato pequeno. Mas o efeito **se mantém** em todos os passos seguintes: o carro vai continuar andando mais rápido até alguém frear. Isso é diferente de virar (efeito imediato no ângulo) e dos sensores (refletem o estado atual).

### 2. Cria dilemas de longo prazo

Acelerar dá recompensa imediata maior (mais $\Delta$progresso por passo). Mas se você acelerar antes de uma curva, vai bater na parede e perder $-100$. O agente precisa aprender que **às vezes é certo desacelerar mesmo perdendo progresso imediato**, antecipando a curva. Esse é o **problema clássico de crédito temporal** (*temporal credit assignment*) que torna RL difícil em geral, e que aparece de forma vívida aqui.

### 3. Interage com o ângulo de virada

A virada é em ângulo absoluto ($\theta \pm 30°$), independente da velocidade. Mas o **raio da curva resultante** depende de $v$:

- Velocidade baixa + virada de 30° = curva apertada, raio pequeno.
- Velocidade alta + virada de 30° = curva larga, raio grande.

O agente precisa **coordenar** velocidade e virada para fazer curvas que caibam no corredor da pista. Se acelerar muito antes de uma curva, mesmo virando o carro vai sair pela tangente e bater.

### C.7 Implicações práticas para o seu agente

1. **Não basta aprender a virar — precisa aprender a desacelerar antes de curvas.** Essa é a habilidade mais difícil que o agente vai dominar, e geralmente é a última a emergir no treinamento.
2. **Curvas de aprendizado “boas mas não ótimas”** geralmente refletem que o agente aprendeu a chegar ao fim mas não otimizou velocidade. Anda devagar o tempo todo (seguro), nunca atinge $V_{\max}$. Política funcional, mas conservadora.
3. **Para depurar visualmente:** rode `renderizar_episodio` no `src/visualize.py` para ver o carro correndo a pista no seu terminal.
    - Se o carro **anda na velocidade máxima sempre e bate**: o problema é aprender a frear.
    - Se o carro **anda devagar sempre e nunca bate mas demora muito**: o problema é aprender a acelerar nas retas.
    - Se o carro **acelera nas retas e freia antes das curvas**: parabéns, está bem treinado.
4. **No relatório:** vale comparar a **velocidade média** atingida pelo agente em cada pista. É um indicador de quão “agressiva” é a política aprendida — em pistas com risco (T4.2), você deve observar uma política mais conservadora, que privilegia segurança sobre velocidade.