"""
Visualização do carrinho de corrida.

Funções principais:
    plotar_pista(grid, ax=None) — desenha a pista estática (matplotlib, para relatório)
    renderizar_episodio(env, agente_fn, ...) — animação no terminal
    plotar_campo_progresso(env, save_path) — PNG do campo BFS (matplotlib)

A animação é renderizada diretamente no terminal usando códigos ANSI para
limpar a tela entre frames. Funciona em qualquer terminal moderno (macOS,
Linux, Windows com Terminal). Não requer pillow.

Uso típico:
    from visualize import renderizar_episodio
    renderizar_episodio(env, lambda obs: agente.escolher_acao(obs))
"""

from __future__ import annotations
import math
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from track import PAREDE, ASFALTO, LARGADA, CHEGADA
from env import AmbienteCarro, ANGULOS_RAIOS


# === Caracteres usados na visualização do terminal ===
EMOJI_PAREDE = "🧱"
EMOJI_ASFALTO = "⚪️"
EMOJI_LARGADA = "🟢"
EMOJI_CHEGADA = "🏁"
EMOJI_RASTRO = "🟦"   # célula já percorrida pelo carro

# Carro: símbolo escolhido conforme o ângulo (8 direções)
SIMBOLOS_CARRO = ["➡️", "↘️", "⬇️", "↙️", "⬅️", "↖️", "⬆️", "↗️"]

# Cores para o plot estático (matplotlib)
COR_PAREDE = "#2c3e50"
COR_ASFALTO = "#ecf0f1"
COR_LARGADA = "#27ae60"
COR_CHEGADA = "#e74c3c"


# === Auxiliar: limpa a tela do terminal e move cursor para o topo ===
def _limpar_terminal():
    print("\033[2J\033[H", end="", flush=True)


def _simbolo_carro(theta: float) -> str:
    """Escolhe um emoji direcional para o carro com base em theta."""
    # Normaliza theta para [0, 2pi) e mapeia em 8 direções
    theta_norm = (theta + math.pi / 8) % (2 * math.pi)
    setor = int(theta_norm / (math.pi / 4)) % 8
    return SIMBOLOS_CARRO[setor]


# === Plot estático da pista (matplotlib) ===
def plotar_pista(grid: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Desenha a pista estática como um conjunto de retângulos coloridos."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    h, w = grid.shape
    mapa_cores = {
        PAREDE: COR_PAREDE,
        ASFALTO: COR_ASFALTO,
        LARGADA: COR_LARGADA,
        CHEGADA: COR_CHEGADA,
    }

    for y in range(h):
        for x in range(w):
            cor = mapa_cores[grid[y, x]]
            ret = patches.Rectangle(
                (x, y), 1, 1, facecolor=cor, edgecolor="white", linewidth=0.5
            )
            ax.add_patch(ret)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plotar_campo_progresso(env: AmbienteCarro, save_path: Optional[str] = None):
    """Plota o campo de progresso (resultado do BFS) para visualizar o reward shaping."""
    fig, ax = plt.subplots(figsize=(10, 7))
    plotar_pista(env.grid, ax=ax)

    h, w = env.grid.shape
    for y in range(h):
        for x in range(w):
            if env.campo_progresso[y, x] >= 0:
                ax.text(
                    x + 0.5, y + 0.5, str(env.campo_progresso[y, x]),
                    ha="center", va="center", fontsize=8,
                    color="black", weight="bold",
                )
    ax.set_title("Campo de progresso (BFS a partir da largada)")
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Salvo em {save_path}")
    plt.close(fig)


# === Animação no terminal ===
def _renderizar_frame(env: AmbienteCarro, rastro: set, info_extra: str = "") -> str:
    """
    Constrói uma string com a representação textual do estado atual da pista.
    Retorna a string pronta para imprimir.
    """
    grid = env.grid
    h, w = grid.shape
    c = env.carro
    cy, cx = int(c.y), int(c.x)

    linhas = []
    for y in range(h):
        linha_chars = []
        for x in range(w):
            if (y, x) == (cy, cx):
                # Posição do carro: emoji direcional
                linha_chars.append(_simbolo_carro(c.theta))
            elif (y, x) in rastro:
                # Rastro do trajeto já percorrido
                linha_chars.append(EMOJI_RASTRO)
            else:
                celula = grid[y, x]
                if celula == PAREDE:
                    linha_chars.append(EMOJI_PAREDE)
                elif celula == LARGADA:
                    linha_chars.append(EMOJI_LARGADA)
                elif celula == CHEGADA:
                    linha_chars.append(EMOJI_CHEGADA)
                else:
                    linha_chars.append(EMOJI_ASFALTO)
        linhas.append("".join(linha_chars))

    grade_str = "\n".join(linhas)

    rodape = (
        f"\nPasso: {env.passos}  |  "
        f"v: {c.v:.2f}  |  "
        f"θ: {math.degrees(c.theta):.0f}°  |  "
        f"Progresso: {env.melhor_progresso_atingido}/{env.progresso_max}"
    )
    if info_extra:
        rodape += f"\n{info_extra}"

    return grade_str + rodape


def renderizar_episodio(
    env: AmbienteCarro,
    agente_fn: Callable[[np.ndarray], int],
    fps: int = 8,
    max_steps: Optional[int] = None,
    limpar_tela: bool = True,
    pausa_final: float = 1.5,
):
    """
    Roda um episódio com agente_fn(obs) -> action e renderiza no terminal.

    Args:
        env: ambiente AmbienteCarro (será chamado reset)
        agente_fn: função que recebe observação e retorna ação
        fps: frames por segundo (controla velocidade da animação)
        max_steps: limita o nº de passos (default: env.max_steps)
        limpar_tela: se True, limpa a tela entre frames (animação fluida).
                     Se False, imprime cada frame em sequência (útil para logs).
        pausa_final: tempo (s) que o último frame fica visível.

    Retorna: (recompensa_total, n_passos, info_final)
    """
    obs = env.reset()
    max_steps = max_steps or env.max_steps
    intervalo = 1.0 / fps

    rastro = set()
    reward_total = 0.0
    info_final = {}

    # Frame inicial (estado de largada)
    if limpar_tela:
        _limpar_terminal()
    print(_renderizar_frame(env, rastro, info_extra="Início do episódio"))
    time.sleep(intervalo)

    for t in range(max_steps):
        # Marca a célula atual no rastro ANTES de mover (para registrar onde passou)
        cy, cx = int(env.carro.y), int(env.carro.x)
        rastro.add((cy, cx))

        action = agente_fn(obs)
        obs, r, term, trunc, info = env.step(action)
        reward_total += r

        # Status do passo
        nomes_acoes = {0: "nada", 1: "acelerar", 2: "frear", 3: "esquerda", 4: "direita"}
        info_extra = f"Ação: {nomes_acoes[action]}  |  Reward: {r:+.2f}  |  Total: {reward_total:+.2f}"

        if limpar_tela:
            _limpar_terminal()
        print(_renderizar_frame(env, rastro, info_extra=info_extra))

        if term or trunc:
            info_final = info
            if info.get("chegada"):
                print("\n🏁 Chegou na linha de chegada!")
            elif info.get("colisao"):
                print("\n💥 Colisão!")
            elif trunc:
                print("\n⏱️  Limite de passos atingido.")
            break

        time.sleep(intervalo)

    time.sleep(pausa_final)
    print(f"\nResumo: {env.passos} passos, recompensa total = {reward_total:.2f}")
    return reward_total, env.passos, info_final


if __name__ == "__main__":
    import sys
    pista = sys.argv[1] if len(sys.argv) > 1 else "pistas/pista_01.txt"
    env = AmbienteCarro(pista)

    # Política trivial: acelera 3x e segue em frente
    contador = [0]
    def agente_trivial(obs):
        contador[0] += 1
        return 1 if contador[0] <= 3 else 0

    reward_total, n_passos, info = renderizar_episodio(env, agente_trivial, fps=4)

    # Bônus: gerar PNG do campo de progresso
    plotar_campo_progresso(env, save_path="/tmp/progresso.png")
