# EP Carrinho — Starter Code

Este pacote contém a infraestrutura básica para o EP do carrinho de corrida com Aprendizado por Reforço Tabular. Ele cobre as partes "chatas" da implementação (parser de pistas, ambiente do carro com física, ray casting, visualização) para que você possa focar no que interessa: **implementar Q-Learning e SARSA**.

## Estrutura do pacote

```
carrinho/
├── README.md                ← este arquivo
├── solucao.py               ← esqueleto a ser preenchido com Q-Learning e SARSA
├── src/
│   ├── track.py             ← parser de pistas em emojis
│   ├── env.py               ← ambiente AmbienteCarro (física + LIDAR + recompensas)
│   └── visualize.py         ← geração de GIF/MP4 do agente correndo
├── pistas/
│   ├── pista_01.txt … pista_08.txt   ← 8 pistas básicas (sanity, retas, curvas suaves)
│   ├── pista_09.txt, pista_10.txt    ← 2 pistas inéditas (uso pedagógico — ver enunciado)
│   └── pista_11.txt … pista_18.txt   ← 8 pistas complexas (zigzags, U-turns, cobras, escadas)
└── tests/
    └── validar_pistas.py    ← valida que todas as pistas têm caminho largada → chegada
```

As pistas estão organizadas por dificuldade crescente:

- **01–02:** retas simples, ideais para depurar a implementação.
- **03–08:** curvas suaves e variações moderadas — boas para os experimentos principais.
- **09–10:** mais complexas, inéditas (use conforme o enunciado pede).
- **11–18:** **complexas com várias mudanças de direção** — zigzags densos, U-turns, formato cobra, escadas diagonais. Para experimentos opcionais ou para investigar limites do tabular.

## Setup

```bash
pip install numpy matplotlib tqdm
```

Não há dependências de imagem (sem `pillow`, sem `PIL`). A animação do agente acontece **diretamente no terminal** com emojis. Matplotlib é usado apenas para gerar PNGs estáticos (campo de progresso, curvas de aprendizado para o relatório).

## Verificando o starter code

Antes de começar a implementar, rode:

```bash
# Valida todas as pistas
python tests/validar_pistas.py

# Testa o ambiente com agente trivial (acelera 3x e segue reto)
PYTHONPATH=src python src/env.py pistas/pista_01.txt

# Gera GIF e campo de progresso (em /tmp por padrão)
PYTHONPATH=src python src/visualize.py pistas/pista_01.txt
```

Se todas as três rodarem sem erro, o ambiente está pronto.

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

O estado é um vetor de 6 floats normalizados em [0, 1] (5 sensores LIDAR + velocidade). Para **Q-Learning e SARSA tabulares**, você precisa **discretizar** esse vetor antes de usar como chave da tabela. A estratégia de discretização afeta muito o desempenho — documente sua escolha no relatório.

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

- **Pista 01–02 (retas):** Q-Learning tabular converge rápido (poucos milhares de episódios).
- **Pista 03 (curva suave):** o baseline do enunciado — pode precisar **20.000–30.000 episódios** com K=5 para uma boa política.
- **Pistas 11–18 (complexas):** podem ser difíceis ou impossíveis para tabular — esperado pedagogicamente. Use para discussão crítica no relatório.

A calibração final dos hiperparâmetros é parte do EP — você vai precisar experimentar.

## Salvamento de modelos

Treinar 30.000 episódios pode demorar minutos. Para evitar re-treinar a cada execução, salve a tabela $Q$ via `pickle` no diretório `/treinamento/`. O `solucao.py` já tem a função utilitária `treinar_ou_carregar()` pronta para isso.

Detalhes no **Anexo B do enunciado**.

## Modificando o ambiente

Arquivos em `src/env.py` que você **pode** ajustar (e documentar no relatório):

- `V_MAX`, `V_DELTA`: velocidade máxima e incremento por aceleração
- `THETA_DELTA`: ângulo por virada (atualmente 30°)
- `DIST_MAX_RAIO`, `N_RAIOS`, `ANGULOS_RAIOS`: configuração dos sensores LIDAR
- `R_TEMPO`, `R_COLISAO`, `R_CHEGADA`: pesos da recompensa

Mudar esses valores muda o problema. Justifique no relatório.

## Esqueleto da sua implementação

Veja `solucao.py` — ele tem placeholders para os dois algoritmos (`AgenteQLearning`, `AgenteSARSA`) e a função `main()` que orquestra a I/O esperada pelo enunciado.

## Dúvidas

- Algo que não roda? Confira `tests/validar_pistas.py` primeiro.
- Política aprendida bate na parede no primeiro passo? Verifique se você está discretizando obs corretamente e usando a chave certa para indexar Q.
- Curva de recompensa fica plana em −100? O agente nunca chega ao fim e episódios sempre terminam em colisão. Aumente `max_steps`, ajuste o schedule de ε, ou comece em pista mais simples.

Bons treinos!
