# Design das 18 pistas — versão ampliada

## Pedido do usuário
- Desenhar 18 pistas com curvas (abertas, fechadas, à esquerda, à direita) levando até a chegada.
- 3 níveis de dificuldade (6 pistas por nível).
- NÃO usar pistas antigas como referência (foram deletadas).
- Pistas precisam ser **grandes** — esta é a 3ª iteração de tamanho, ainda maior que as anteriores.
- Difíceis devem combinar várias curvas.

## Formato dos arquivos
- Emojis separados por espaços; linhas separadas por `\n`.
- `🧱` = parede, `⚪` = asfalto, `🟢` = largada, `🏁` = chegada.
- Carro larga voltado para leste; primeira manobra deve ser viável.
- Carro: V_MAX=2.0, vira 30° por ação, LIDAR alcance 10 células.

## Dimensões adotadas (escala grande)

| Nível | Largura | Altura | Corredor |
|---|---|---|---|
| Fácil (01–06) | 28–32 col | 18–22 lin | 5 células |
| Médio (07–12) | 38–46 col | 26–32 lin | 4 células |
| Difícil (13–18) | 50–62 col | 36–42 lin | 3 células (com trechos de 4) |

## NÍVEL 1 — FÁCIL (corredor largo, 1–2 curvas abertas)

### pista_01 — "L grande à direita" (~30×20)
Reta horizontal longa para leste (~22 colunas), curva ampla de 90° para sul (raio ~4), reta vertical descendo (~10 linhas). Corredor 5 células.

### pista_02 — "L grande à esquerda" (~30×20)
Espelho da 01. Reta longa para leste, curva ampla 90° para norte (subindo), reta vertical.

### pista_03 — "S aberto longo" (~32×20)
Reta leste, curva ampla descendo, trecho horizontal médio, curva ampla voltando à horizontal, reta final. Duas curvas abertas opostas.

### pista_04 — "Arco contínuo amplo" (~30×22)
Curva única descrevendo ~120° de arco aberto, raio grande. Larga indo leste, termina indo sul-sudoeste.

### pista_05 — "U aberto grande" (~32×22)
Reta leste, curva ampla descendo, reta horizontal embaixo (volta no sentido inverso), curva ampla para oeste, reta final. U deitado bem aberto.

### pista_06 — "Triângulo arredondado" (~30×22)
Três trechos retos conectados por duas curvas abertas de ~45° (leste → sudeste → sul).

## NÍVEL 2 — MÉDIO (corredor 4, 4–6 curvas)

### pista_07 — "Z médio com chicane" (~40×28)
Reta leste, 90° fechada para sul, reta vertical, 90° fechada para leste, chicane suave embutida na reta final. ~5 curvas.

### pista_08 — "Onda quádrupla" (~42×26)
Quatro curvas alternadas em sequência (S-S-S), retas curtas entre elas. Corredor 4.

### pista_09 — "Três-quartos de circuito" (~38×30)
~270° de giro (leste → sul → oeste → norte). Três curvas de 90° fechadas, retas amplas entre elas.

### pista_10 — "Escada diagonal longa" (~44×30)
Caminho em degraus: 4 degraus horizontais + 4 trechos verticais curtos. 8 curvas de 90° alternadas.

### pista_11 — "L duplo + chicane" (~40×30)
Reta leste, 90° para sul, vertical, chicane apertada (S) no meio, 90° para leste, reta final.

### pista_12 — "P deitado (U-turn)" (~40×30)
Reta leste, 90° para norte (sobe), U-turn 180° no topo, desce paralelamente, 90° para leste, reta final.

## NÍVEL 3 — DIFÍCIL (corredor 3, 8+ curvas combinadas, grandes)

### pista_13 — "Zigue-zague longo combinado" (~58×24)
8 curvas alternadas fechadas + curva final de 90°. Trechos retos curtos (~3 colunas). Corredor 3.

### pista_14 — "Duplo U-turn com chicane" (~52×38)
Reta inicial, U-turn 180° apertado, reta paralela, chicane apertada (S+S), outro U-turn 180°, reta final.

### pista_15 — "Espiral retangular completa" (~50×40)
Espiral 1.5 voltas em sentido horário até o centro. 6 curvas de 90° fechadas em sequência. Chegada no miolo.

### pista_16 — "Chicane quíntupla + curva final" (~56×32)
5 S-curves apertados consecutivos (10 curvas), corredor 3, fecha com 90° fechada final.

### pista_17 — "Circuito misto grande" (~52×42)
Circuito quase-fechado horário: 4 curvas 90° fechadas + chicane no trecho leste + U-turn interno no trecho sul. ~10 curvas combinadas.

### pista_18 — "Serpente combinada gigante" (~62×40)
Pista enorme: 6 curvas alternadas + 1 U-turn meio + 1 chicane tripla + 2 curvas 90° fechadas no final. Mistura todos os elementos. Trechos largos (4) e estreitos (3).
