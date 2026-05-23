# Entrega e Avaliação

## Grupo

O EP pode ser feito **individualmente ou em grupos de até 3 pessoas**. Todos os integrantes precisam dominar a modelagem e o código — qualquer um pode ser sorteado para apresentar.

## Código fonte

Deverá ser entregue o **repositório no GitHub**. Deve ser implementado usando o ambiente de testes em https://github.com/senac-ia/rf-carro-autonomo

Não é obrigatório, mas preferencialmente usar Python. A correção do professor considera que o algoritmo deve rodar em sua máquina local, portanto, deverá ter as instruções de como rodar e dependências no `README.md`.

**Não será permitido o uso de bibliotecas de Inteligência Artificial pronta** (`gymnasium`, `stable-baselines3`, `ray[rllib]`, `tianshou`, etc.). Você deve implementar o Q-Learning **do zero**. Você pode usar `numpy` e `tqdm` (já listados em `requirements.txt`).

Pode ser baseado no starter code fornecido em https://github.com/senac-ia/rf-carro-autonomo que contém:

- Parser de pistas em emojis (`src/track.py`)
- Ambiente do carro com física e LIDAR (`src/env.py`)
- Visualização animada (`src/visualize.py`)
- 18 pistas (`pistas/pista_01.txt` a `pistas/pista_18.txt`)
- Esqueleto de `solucao.py` para preencher

## Salvamento de modelos treinados

Como os tempos de treinamento podem ser longos (30.000 episódios em pistas mais complexas leva vários minutos), você **deve salvar os modelos treinados em disco**, no diretório `/treinamento` na raiz do projeto. Use `pickle` da biblioteca padrão do Python (ver [`anexo_b_pickle.md`](anexo_b_pickle.md)).

Estrutura esperada na raiz do projeto:

```
seu-projeto/
├── README.md
├── solucao.py
├── src/
├── pistas/
└── treinamento/
    └── q_learning_K5_pista_03.pkl  ← Q-Learning treinado em pista_03
```

Esse arquivo deve ser **commitado no repositório**. Isso permite ao professor reproduzir a avaliação sem re-treinar.

## O que será avaliado

### Em apresentação

- **Representação do espaço de estados:**
    - Como você implementou a discretização do vetor de 6 floats com $K = 5$?
    - Qual o tamanho real da tabela $Q$ ao final do treinamento? Como esse número se compara ao máximo teórico de $5^6 \times 5 = 78{.}125$ entradas?
- **Espaço de ações:** como as 5 ações foram codificadas?
- **Função de recompensa:** como o reward shaping foi implementado? Você experimentou variações?
- **Política de exploração:** schedule de $\varepsilon$, justificativa.

> **Atenção:** fazer cópia do algoritmo apenas e explicar o que é o conceito **não vale**. O trabalho requer a explicação de como o conceito foi **modelado e implementado para este problema específico** (pilotar um carrinho).

### Em código

- **Implementação do Q-Learning** do zero.
- **Função de discretização** com $K = 5$.
- **Loop de treinamento** que registra histórico de recompensas por episódio (salvo no pickle do modelo).
- **Loop de avaliação** com $\varepsilon = 0$ que gera o arquivo de saída `q_learning.txt`.
- **Inspeção da política final** via animação no terminal (`renderizar_episodio` em `src/visualize.py`) — descreva no relatório o que observou para a política treinada em cada pista.
- **Salvamento e carregamento** dos modelos via pickle no diretório `/treinamento`.

### Critérios de avaliação

- Explicação da lógica do problema e da modelagem do MDP.
- Explicação da discretização adotada (por que $K = 5$ funciona bem aqui).
- Explicação das funções principais e estrutura do código.
- Demonstração dos resultados (histórico de aprendizado em formato textual, animação do agente no terminal, tabelas de avaliação).
- **Análise crítica** do comportamento do Q-Learning na pista (curva de aprendizado, política aprendida, trajetória observada).
- Criatividade — extensões além do mínimo, exploração de variações na função de recompensa ou em outras pistas.

## Política de uso de ferramentas

Este trabalho deve seguir:

- [Política de uso de ferramentas generativas de IA](https://www.notion.so/...)
- [Política antiplágio](https://www.notion.so/...)
