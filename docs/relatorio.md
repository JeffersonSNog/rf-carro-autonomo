# Relatório — EP Carro Autônomo: Q-Learning Tabular

**Disciplina:** Inteligência Artificial  
**Instituição:** Centro Universitário Senac SP  
**Aluno(s):** João Pedro Ferreira da Rocha, Jefferson Santana Nogueira, Lucas de Pinho Ribeiro dos Santos
**Data:** Junho de 2026

---

## 1. Modelagem do MDP

### 1.1 Estados

O estado observável é um vetor de 6 floats em [0, 1]:

```
obs = [d_0, d_+30, d_-30, d_+60, d_-60, v_norm]
```

- **d_α** (5 sensores LIDAR): distância normalizada até a parede mais próxima na direção θ+α, dividida por DIST_MAX_RAIO = 10 células. Saturada em 1,0 quando não há parede em até 10 células.
- **v_norm**: velocidade atual normalizada por V_MAX = 2,0. Como v ∈ {0; 0,5; 1,0; 1,5; 2,0}, temos v_norm ∈ {0,00; 0,25; 0,50; 0,75; 1,00}.

O agente não conhece sua posição absoluta nem sua orientação no mapa — só o que os sensores enxergam à frente. Isso torna o aprendizado generalizável entre pistas distintas: padrões como "parede frontal próxima + corredor abre à direita → vire" transferem para qualquer pista com geometria local semelhante.

### 1.2 Ações

Espaço discreto de 5 ações:

| Código | Ação             | Efeito                  |
| ------ | ---------------- | ----------------------- |
| 0      | Nada             | Mantém v e θ            |
| 1      | Acelerar         | v ← min(v + 0,5, V_max) |
| 2      | Frear            | v ← max(v − 0,5, 0)     |
| 3      | Virar à esquerda | θ ← θ − 30°             |
| 4      | Virar à direita  | θ ← θ + 30°             |

### 1.3 Recompensas

| Evento                     | Recompensa              |
| -------------------------- | ----------------------- |
| Avanço de progresso (Δs)   | +Δs por passo           |
| Custo de tempo             | −0,1 por passo          |
| Colisão com parede         | −100 (episódio termina) |
| Chegada à linha de chegada | +500 (episódio termina) |

O reward shaping via progresso é fundamental: sem ele, a recompensa seria esparsa (+500 só na chegada) e o agente levaria muito mais episódios para aprender qualquer política útil. O Δs guia o agente na direção certa desde os primeiros episódios, mesmo antes de completar qualquer pista.

---

## 2. Discretização do Estado

Como Q-Learning tabular exige estados discretos, convertemos o vetor de 6 floats em uma tupla de 6 inteiros usando **binning uniforme com K=5**:

```python
def discretizar(obs, K=5):
    return tuple(min(int(v * K), K - 1) for v in obs)
```

**Exemplo** (obs = [0.35, 1.00, 0.30, 0.41, 0.18, 0.50]):

| Componente | v    | v×5  | int(·) | min(·, 4)     |
| ---------- | ---- | ---- | ------ | ------------- |
| d_0        | 0,35 | 1,75 | 1      | **1**         |
| d\_+30     | 1,00 | 5,00 | 5      | **4** (clamp) |
| d\_-30     | 0,30 | 1,50 | 1      | **1**         |
| d\_+60     | 0,41 | 2,05 | 2      | **2**         |
| d\_-60     | 0,18 | 0,90 | 0      | **0**         |
| v_norm     | 0,50 | 2,50 | 2      | **2**         |

→ chave = **(1, 4, 1, 2, 0, 2)**

### Por que K=5?

1. **Casa com a granularidade da velocidade**: os 5 valores físicos de v_norm (0,00; 0,25; 0,50; 0,75; 1,00) caem em 5 baldes distintos, sem perda de informação.
2. **Resolução adequada para o LIDAR**: cada balde cobre 2 células (20% do alcance), suficiente para distinguir "colado na parede" de "com folga".
3. **Tabela manejável**: 5⁶ = 15.625 estados possíveis. Ao final do treinamento, apenas **6.463 estados foram populados** — menos da metade do espaço teórico, confirmando que o defaultdict foi a escolha certa.

### Estrutura da tabela Q

```python
from collections import defaultdict
self.Q = defaultdict(lambda: np.zeros(self.n_actions))
```

Inicializa estados on-demand, alocando memória apenas para estados realmente visitados.

---

## 3. Implementação do Q-Learning

### 3.1 Atualização TD (off-policy)

```
alvo = r + γ · max_a' Q(s', a')   se não terminou
alvo = r                            se terminou

Q(s, a) ← Q(s, a) + α · (alvo − Q(s, a))
```

O Q-Learning é **off-policy**: aprende sobre a política gulosa (argmax Q) mesmo enquanto explora com ε-greedy. O `max` no alvo é independente da ação que será tomada no próximo passo.

### 3.2 Política ε-greedy

```python
if random.random() < self.eps:
    return random.randrange(self.n_actions)   # exploração
return int(np.argmax(self.Q[chave]))          # explotação
```

### 3.3 Esquema de treinamento round-robin

A cada episódio, uma pista do conjunto de treino (01–16) é sorteada aleatoriamente:

```python
pista = random.choice(pistas_treino)
env   = envs[pista]  # cache — não recalcula BFS
```

Isso evita **catastrophic forgetting**: treinar sequencialmente faria o agente desaprender as primeiras pistas ao chegar nas últimas. Com round-robin, todos os padrões são reforçados ao longo de todo o treinamento.

---

## 4. Hiperparâmetros

### 4.1 Taxa de aprendizado α = 0,15

Valor ligeiramente acima do padrão de Sutton & Barto (0,1) para compensar a diversidade de pistas no round-robin. α=0,1 convergiu mais lentamente nas pistas médias (05–12); α=0,2 apresentou oscilação nas pistas fáceis. α=0,15 equilibrou velocidade de aprendizado e estabilidade.

### 4.2 Fator de desconto γ = 0,97

Valor um pouco abaixo do sugerido (0,99). γ=0,99 faz o agente supervalorizar recompensas muito distantes e dificulta aprender a frear antes de curvas. γ=0,97 equilibra planejamento de longo prazo com responsividade a recompensas imediatas.

**Horizonte de planejamento efetivo:** após 66 passos, γ⁶⁶ ≈ 0,13 — o agente efetivamente "enxerga" ~66 passos à frente com peso significativo.

### 4.3 Política ε-greedy

| Parâmetro        | Valor                           |
| ---------------- | ------------------------------- |
| ε inicial        | 1,0 (totalmente aleatório)      |
| ε final          | 0,05 (95% guloso)               |
| Schedule         | Linear em 80% dos episódios     |
| ε nos 20% finais | 0,05 fixo (exploração residual) |

O decaimento linear foi escolhido pela simplicidade e previsibilidade. Os 20% finais com ε=0,05 mantêm exploração residual para evitar convergência para mínimos locais.

**Ponto de transição:** com 480.000 episódios totais, o decaimento ocorre nos primeiros 384.000 episódios. A partir do episódio 384.001, ε=0,05 fixo.

### 4.4 Orçamento de treino

| Parâmetro                    | Valor                 |
| ---------------------------- | --------------------- |
| Episódios por pista          | 30.000                |
| Pistas de treino             | 16 (01–16)            |
| **Total**                    | **480.000 episódios** |
| Limite de passos (max_steps) | 500                   |
| Estados populados ao final   | **6.463**             |

---

## 5. Mecânica da Exploração

Durante o treino, a escolha de ação funciona assim:

```
1. Gerar número aleatório u ~ Uniforme(0, 1)
2. Se u < ε:
       → ação aleatória uniforme em {0, 1, 2, 3, 4}
   Senão:
       → chave = discretizar(obs)
       → ação = argmax Q[chave]
```

**Por que ε alto no início é crítico:** no início, Q está zerado. argmax de zeros retorna sempre a ação 0 (nada) — o carro ficaria parado. ε=1,0 garante que todas as ações sejam exploradas e o agente descubra que acelerar + virar leva a progresso positivo.

---

## 6. Curva de Aprendizado

A evolução do treinamento mostra três fases claras:

| Fase            | Episódios | R̄          | Sucesso | Interpretação                               |
| --------------- | --------- | ---------- | ------- | ------------------------------------------- |
| Exploração pura | 0–110k    | ~−96 a −92 | 0,0%    | Agente só explora, aprende estrutura básica |
| Transição       | 110k–300k | −92 a −35  | 0%–7%   | Primeiros sucessos esporádicos              |
| Convergência    | 300k–384k | −35 a +333 | 7%–61%  | ε decaindo, política gulosa emerge          |
| Refinamento     | 384k–480k | ~263–352   | 51%–63% | ε=0,05 fixo, política se consolida          |

### Resumo por pista (últimos 500 episódios de treino)

| Pista    | R̄     | Nível   |
| -------- | ----- | ------- |
| pista_01 | 449,8 | Fácil   |
| pista_02 | 365,0 | Fácil   |
| pista_03 | 434,8 | Fácil   |
| pista_04 | 403,8 | Fácil   |
| pista_05 | 373,6 | Médio   |
| pista_06 | 381,5 | Médio   |
| pista_07 | 370,2 | Médio   |
| pista_08 | 186,4 | Médio   |
| pista_09 | 357,7 | Médio   |
| pista_10 | 359,0 | Médio   |
| pista_11 | 302,3 | Médio   |
| pista_12 | 270,1 | Médio   |
| pista_13 | 144,4 | Difícil |
| pista_14 | −1,8  | Difícil |
| pista_15 | 198,5 | Difícil |
| pista_16 | 118,2 | Difícil |

---

## 7. Resultados nas Pistas de Holdout

### 7.1 Pista 17 — "Circuito misto com gargalo"

```
=== Pista: pista_17.txt ===
Algoritmo: Q-Learning (round-robin em pistas 01-16)
Episódios totais de treinamento: 480000
Estados populados: 6463
Tempo de chegada (passos): 357
Velocidade média: 1.246
Velocidade máxima atingida: 2.000
Recompensa total: 603.30
Sucesso: SIM
```

### 7.2 Pista 18 — "Serpente combinada"

```
=== Pista: pista_18.txt ===
Algoritmo: Q-Learning (round-robin em pistas 01-16)
Episódios totais de treinamento: 480000
Estados populados: 6463
Tempo de chegada (passos): 232
Velocidade média: 0.899
Velocidade máxima atingida: 2.000
Recompensa total: 623.80
Sucesso: SIM
```

### 7.3 Comparação treino vs. holdout

| Conjunto                | R̄ (últimos 500 ep) | Resultado  |
| ----------------------- | ------------------ | ---------- |
| Fáceis (01–04)          | ~413               | Excelente  |
| Médias (05–12)          | ~300               | Bom        |
| Difíceis treino (13–16) | ~115               | Parcial    |
| **Pista 17 (holdout)**  | **603,30**         | **✅ SIM** |
| **Pista 18 (holdout)**  | **623,80**         | **✅ SIM** |

Notavelmente, as pistas de holdout tiveram **recompensa superior** à média das difíceis de treino (13–16). Isso indica que a pista_17 e pista_18, apesar de serem classificadas como difíceis, têm geometria local que combina bem com os padrões LIDAR aprendidos nas 16 pistas de treino.

---

## 8. Análise Crítica de Generalização

### 8.1 O que o LIDAR permite generalizar

O LIDAR é **local e relativo ao carro**: o agente vê distâncias à frente e nos lados, mas não sabe onde está no mapa absoluto. Padrões de comportamento aprendidos em uma pista transferem para qualquer pista com geometria local semelhante.

Por exemplo, ao aprender em pistas com curvas à direita, o agente associa o padrão (d*0 pequeno, d*+30 grande) à ação de virar à direita. Esse padrão é **geométrico, não posicional** — funciona em qualquer pista com curvas similares.

### 8.2 Por que as holdouts foram bem?

A pista_17 ("Circuito misto") combina loop externo + chicane + U-turn — todos elementos presentes nas pistas 05–16 do treino. A pista_18 ("Serpente combinada") é descrita como "teste final de todos os elementos aprendidos", e o agente passou com recompensa 623,80.

O sucesso em ambas confirma que a representação LIDAR + round-robin em 16 pistas produziu uma política genuinamente generalizável, não apenas memorizada.

### 8.3 Limitações do Q-Learning tabular

**Sem interpolação:** Q tabular não generaliza entre estados vizinhos. Um estado (1,4,1,2,0,2) e (1,4,1,2,1,2) são completamente independentes — mesmo sendo fisicamente quase idênticos.

**Pista_14 com R̄=−1,8:** a pista "Duplo U-turn + chicane corredor 2" foi a mais difícil do treino. O corredor 2 exige velocidade muito baixa e o agente teve dificuldade em aprender a frear com antecedência suficiente — dilema clássico de crédito temporal descrito no Anexo C do enunciado.

**Estados não visitados:** apenas 6.463 dos 15.625 estados possíveis foram populados. Em geometrias muito incomuns, o agente pode encontrar estados com Q=0 e agir de forma degenerada.

---

## 9. Como Reproduzir

```bash
# Instalar dependências
pip install -r requirements.txt

# Treinar e avaliar (gera pkl + txts de holdout)
python solucao.py

# Forçar re-treino
python solucao.py --recarregar

# Avaliar apenas uma pista (requer pkl existente)
python solucao.py --avaliar pistas/pista_17.txt

# Visualizar o agente no terminal
PYTHONPATH=src python src/visualize.py pistas/pista_17.txt
PYTHONPATH=src python src/visualize.py pistas/pista_18.txt
```

---

## 10. Referências

- Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_ (2ª ed.). MIT Press. Disponível em: http://incompleteideas.net/book/the-book.html
- Watkins, C. J. C. H. (1989). _Learning from Delayed Rewards_. Tese de doutorado, University of Cambridge.
- Enunciado do EP: `README.md` e documentos em `enunciado/` (qlearning.md, discretizacao.md, anexo_a_lidar.md, anexo_b_pickle.md, anexo_c_velocidade.md).
