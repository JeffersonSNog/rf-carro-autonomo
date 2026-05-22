# EP Carrinho — Starter Code

Este pacote contém a infraestrutura básica para o EP do carrinho de corrida com Aprendizado por Reforço Tabular. Ele cobre as partes "chatas" da implementação (parser de pistas, ambiente do carro com física, ray casting, visualização) para que você possa focar no que interessa: **implementar o Q-Learning**.

## Estrutura do pacote

```
carrinho/
├── README.md                ← este arquivo
├── solucao.py               ← esqueleto a ser preenchido com Q-Learning
├── src/
│   ├── track.py             ← parser de pistas em emojis
│   ├── env.py               ← ambiente AmbienteCarro (física + LIDAR + recompensas)
│   └── visualize.py         ← geração de GIF/MP4 do agente correndo
├── pistas/
│   ├── pista_01.txt … pista_04.txt   ← 4 pistas FÁCEIS (progressão pedagógica)
│   ├── pista_05.txt … pista_12.txt   ← 8 pistas MÉDIAS (combinam vários elementos)
│   └── pista_13.txt … pista_18.txt   ← 6 pistas DIFÍCEIS (corredor até 2 células)
└── tests/
    └── validar_pistas.py    ← valida que todas as pistas têm caminho largada → chegada
```

As pistas estão organizadas em três níveis de dificuldade (ver `descricao_pistas.md` para o design detalhado):

- **01–04 (fáceis):** progressão pedagógica — cada uma introduz uma habilidade nova (reagir a parede frontal, generalizar curvas, ajuste fino de ângulo, U-turn com chicane). Corredor 3–4 células.
- **05–12 (médias):** combinam vários elementos (chicanes, curvas em sequência, mudanças de direção). Corredor 3–4 células. A `pista_07.txt` é a usada na T4.2 (Cliff-style).
- **13–18 (difíceis):** corredor pode chegar a 2 células, com várias mudanças de direção. Para experimentos opcionais ou discussão de limites do tabular no relatório.

## Setup

```bash
pip install -r requirements.txt
```

**Bibliotecas proibidas:** `gymnasium`, `stable-baselines3`, `ray[rllib]`, `tianshou`, `torch`, ou qualquer biblioteca de RL pronta. O Q-Learning deve ser implementado **do zero**, incluindo a função de discretização. Veja §7 do enunciado.

## Verificando o starter code

Antes de começar a implementar, rode:

```bash
# Valida todas as pistas
python tests/validar_pistas.py

# Testa o ambiente com agente trivial (acelera 3x e segue reto)
PYTHONPATH=src python src/env.py pistas/pista_01.txt

# Anima um episódio no terminal
PYTHONPATH=src python src/visualize.py pistas/pista_01.txt
```

Se todas as três rodarem sem erro, o ambiente está pronto.

## O ambiente de simulação

O `AmbienteCarro` (`src/env.py`) simula um carrinho 2D que precisa percorrer uma pista do ponto de **largada** até a **chegada** sem bater nas paredes. É inspirado na API do Gymnasium (`reset`/`step`) mas implementado do zero, sem dependências externas além de `numpy`.

### Pista

A pista é uma grade 2D carregada de um arquivo de emojis (`pistas/pista_XX.txt`), com quatro tipos de célula: **parede**, **asfalto**, **largada** e **chegada**. Em paralelo, é pré-computado um **campo de progresso** via BFS — cada célula pilotável recebe sua distância em passos até a largada. Esse campo é o que permite o reward shaping.

### Estado do carro (físico, interno)

O carro tem quatro variáveis contínuas: posição `(x, y)` (coordenadas em células, não inteiros), ângulo `θ` (em radianos, `0` = leste, `π/2` = sul) e velocidade `v` (células por passo). A física é simples: a cada passo, a posição é atualizada por `Δx = v·cos(θ)`, `Δy = v·sin(θ)`.

### Observação (o que o agente vê)

A observação retornada por `reset()` e `step()` é um vetor de **6 floats normalizados em [0, 1]**:

- **5 sensores LIDAR** (raios lançados a 0°, ±30°, ±60° em relação à direção do carro), medindo distância até a parede mais próxima, normalizada pelo alcance máximo (`DIST_MAX_RAIO = 10` células).
- **Velocidade normalizada** (`v / V_MAX`).

O carro **não conhece sua posição absoluta nem sua orientação na pista** — só o que os sensores enxergam à frente.

### Ações

Cinco ações discretas:

| Ação | Efeito |
|---|---|
| `0` | nada |
| `1` | acelerar (`+V_DELTA`, limitado por `V_MAX`) |
| `2` | frear (`−V_DELTA`, mínimo 0) |
| `3` | virar à esquerda (`−THETA_DELTA = −30°`) |
| `4` | virar à direita (`+THETA_DELTA = +30°`) |

### Recompensa (reward)

A cada passo, o agente recebe:

- `R_TEMPO = −0.1` (custo de tempo, incentiva terminar rápido)
- `+Δprogresso`: ganho equivalente ao **avanço** no campo de progresso BFS (só conta progresso novo — voltar pelo mesmo lugar não rende nada)
- `R_COLISAO = −100` ao bater numa parede (encerra o episódio)
- `R_CHEGADA = +500` ao atingir a linha de chegada (encerra o episódio)

### Fim de episódio

Um episódio termina (`terminated = True`) por **colisão** ou **chegada**, ou é truncado (`truncated = True`) ao atingir `max_steps` (padrão 500).

## API do ambiente

```python
from env import AmbienteCarro

env = AmbienteCarro("pistas/pista_01.txt", max_steps=500, seed=42)

obs = env.reset()              # vetor de 6 floats: [d_0, d_+30, d_-30, d_+60, d_-60, v_norm]
print(env.obs_dim)             # 6
print(env.n_actions)           # 5

# Loop básico
done = False
while not done:
    action = sua_politica(obs)            # 0=nada, 1=acel, 2=frear, 3=esq, 4=dir
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    # info pode ter {"chegada": True}, {"colisao": True}, ou {}
```

> 💡 **Sobre os nomes:** termos canônicos de Aprendizado por Reforço (`reset`, `step`, `obs`, `action`, `reward`, `terminated`, `truncated`, `info`) são mantidos em inglês para alinhamento com a literatura (Sutton & Barto, Gymnasium). Tudo mais está em português: `AmbienteCarro`, `escolher_acao`, `treinar`, `avaliar`, `discretizar`, etc.

## Pontos importantes que você precisa saber

### 1. Estado é baixo-dimensional, mas **contínuo**

O estado é um vetor de 6 floats normalizados em [0, 1] (5 sensores LIDAR + velocidade). Para o **Q-Learning tabular**, você precisa **discretizar** esse vetor antes de usar como chave da tabela. O enunciado fixa $K = 5$ baldes por dimensão; o §2.1 explica por que essa escolha funciona bem para este problema.

> 💡 **Velocidade é o componente sutil.** Os 5 sensores LIDAR refletem o estado atual; já a velocidade tem efeito acumulativo (acelerar agora afeta todos os passos seguintes) e cria o clássico problema de *temporal credit assignment* — o agente precisa aprender a frear ANTES de uma curva, sacrificando recompensa imediata. Leia o **Anexo C** (`docs/enunciado.md`) para a discussão completa.

### 2. Reward shaping já está implementado

O ambiente calcula um campo de progresso por BFS a partir da largada. A cada passo, você recebe `+Δprogresso` (variação no melhor progresso já alcançado) — isso ajuda muito o aprendizado em comparação a recompensa esparsa pura.

Se quiser experimentar com **recompensa esparsa** (apenas chegada/colisão), modifique `env.py` na função `step`.

### 3. Visualização

A função `renderizar_episodio` no `src/visualize.py` recebe seu agente treinado e mostra o carro correndo a pista **diretamente no seu terminal**, com animação fluida via códigos ANSI (limpa a tela entre frames). Use isso para depuração — ver o agente em ação revela bugs que números não revelam:

```python
from visualize import renderizar_episodio
import numpy as np

def politica(obs):
    # sua política treinada com ε=0
    chave = agente.discretizar(obs)
    return int(np.argmax(agente.Q[chave]))

reward_total, n_passos, info = renderizar_episodio(env, politica, fps=8)
```

O carro é representado por uma seta direcional (➡️ ⬇️ ⬅️ ⬆️ etc.) que muda conforme o ângulo. As células já percorridas ficam azuis (🟦), facilitando ver a trajetória.

## Conta-gotas de viabilidade

Para você ter referência sobre o que esperar:

- **Pistas 01–04 (fáceis):** Q-Learning tabular converge rápido (poucos milhares de episódios).
- **Pista 03 (curva suave):** o baseline do enunciado — pode precisar **20.000–30.000 episódios** com $K=5$ para uma boa política.
- **Pista 07 (curva apertada):** a usada na T4.2 (Cliff-style) — alto risco de colisão durante exploração.
- **Pistas 05–12 (médias):** combinam mais elementos, exigem políticas mais sofisticadas.
- **Pistas 13–18 (difíceis):** corredor mínimo de 2 células com várias mudanças de direção — podem ser muito difíceis ou impossíveis para tabular. Use para discussão crítica no relatório.

A calibração final dos hiperparâmetros é parte do EP — você vai precisar experimentar.

## Tarefas a implementar

Antes de começar, leia **`docs/qlearning.md`** — explica a matemática do Q-Learning (atualização TD, $\varepsilon$-greedy, por que é off-policy), traz o pseudocódigo e dicas de implementação em Python com a estrutura de dados sugerida para a tabela $Q$.

O EP é dividido em duas tarefas (detalhes em `docs/enunciado.md`, §4):

- **T4.1 — Q-Learning Baseline** em `pista_03.txt` com $K=5$. Treine, avalie com $\varepsilon=0$ e gere `q_learning.txt`.
- **T4.2 — Q-Learning em pista de risco (Cliff-style)** em `pista_07.txt`. Treine com a mesma configuração da T4.1 e analise como o agente se comporta quando colidir custa caro: curva de aprendizado, velocidade média da política final, distância das paredes nas trajetórias. Gere `cliff.txt`.

### Hiperparâmetros padrão (baseline do enunciado)

| Hiperparâmetro | Valor |
|---|---|
| Episódios de treinamento | 30.000 |
| Limite de passos por episódio | 500 |
| Discretização $K$ | 5 baldes por dimensão |
| Taxa de aprendizado $\alpha$ | 0,1 |
| Desconto $\gamma$ | 0,99 |
| Exploração $\varepsilon$ | decai linearmente de 1,0 a 0,05 nos primeiros 80% dos episódios |

Use estes valores como ponto de partida. $K = 5$ é **fixo** neste EP (ver §2.1 do enunciado para a justificativa); justifique no relatório qualquer desvio nos demais hiperparâmetros.

### Formato dos arquivos de saída

Cada tarefa gera um `.txt` na raiz do projeto. Template (§5 do enunciado):

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

Cada arquivo é independente — um para a T4.1 e outro para a T4.2.

## Salvamento de modelos

Treinar 30.000 episódios pode demorar minutos. Para evitar re-treinar a cada execução, salve a tabela $Q$ via `pickle` no diretório `/treinamento/`. O `solucao.py` já tem a função utilitária `treinar_ou_carregar()` pronta para isso.

Estrutura esperada do diretório:

```
treinamento/
├── q_learning_K5_pista_03.pkl   ← T4.1 baseline
└── q_learning_K5_pista_07.pkl   ← T4.2 Cliff-style
```

Cada `.pkl` deve guardar pelo menos: tabela Q, $K$ usado, nº de episódios, hiperparâmetros, seed, pista usada e histórico de recompensas (em janela móvel de 100). Esses arquivos devem ser commitados no repositório — assim o professor reproduz as avaliações sem re-treinar.

Detalhes no **Anexo B do enunciado** (`docs/enunciado.md`).

## Modificando o ambiente

Arquivos em `src/env.py` que você **pode** ajustar (e documentar no relatório):

- `V_MAX`, `V_DELTA`: velocidade máxima e incremento por aceleração
- `THETA_DELTA`: ângulo por virada (atualmente 30°)
- `DIST_MAX_RAIO`, `N_RAIOS`, `ANGULOS_RAIOS`: configuração dos sensores LIDAR
- `R_TEMPO`, `R_COLISAO`, `R_CHEGADA`: pesos da recompensa

Mudar esses valores muda o problema. Justifique no relatório.

## Esqueleto da sua implementação

Veja `solucao.py` — ele tem o placeholder do agente (`AgenteQLearning`) e a função `main()` que orquestra a I/O esperada pelo enunciado.

## Relatório

O relatório deve cobrir duas seções (detalhes em `docs/enunciado.md`, §6):

1. **Modelagem do MDP e Q-Learning Baseline (T4.1)** — espaço de estados após a discretização ($K = 5$), ações, recompensa, estrutura de armazenamento de $Q[s,a]$, e os resultados do baseline (passos até completar, velocidade média, perfil de uso de cada ação).
2. **Q-Learning em pista de risco — Cliff-style (T4.2)** — como o comportamento do Q-Learning muda quando colidir custa caro: curva de aprendizado vs. T4.1, recompensa durante treinamento vs. avaliação gulosa, velocidade média, e diferenças qualitativas nas trajetórias observadas via animação no terminal.

## Dúvidas

- Algo que não roda? Confira `tests/validar_pistas.py` primeiro.
- Política aprendida bate na parede no primeiro passo? Verifique se você está discretizando obs corretamente e usando a chave certa para indexar Q.
- Curva de recompensa fica plana em −100? O agente nunca chega ao fim e episódios sempre terminam em colisão. Aumente `max_steps`, ajuste o schedule de ε, ou comece em pista mais simples.

Bons treinos!
