"""
Solução completa para o EP do carrinho — Q-Learning tabular.

Implementa:
    - AgenteQLearning  (tabular, discretização K-bins, ε-greedy com decaimento)
    - treinar_round_robin (round-robin nas 16 pistas de treino)
    - avaliar (política gulosa)
    - escrever_saida (formato README §4.3)
    - main() (treino → pkl → avaliação → txts)

Uso:
    python solucao.py                         # treina + avalia em pistas 17 e 18
    python solucao.py --recarregar            # força re-treino
    python solucao.py --avaliar pistas/X.txt  # avalia modelo salvo em X
"""

import sys
import random
import argparse
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from env import AmbienteCarro  # noqa: E402


# === Configuração global ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DIR_TREINAMENTO = Path("treinamento")
DIR_TREINAMENTO.mkdir(exist_ok=True)

PISTAS_TREINO  = [f"pistas/pista_{i:02d}.txt" for i in range(1, 17)]   # 01..16
PISTAS_HOLDOUT = [f"pistas/pista_{i:02d}.txt" for i in range(17, 19)]  # 17, 18


# ============================================================================
# Q-LEARNING TABULAR
# ============================================================================

class AgenteQLearning:
    """
    Q-Learning tabular com:
      - Discretização uniforme de K bins por dimensão (K=5 padrão).
        bin_i = min(floor(v_i * K), K-1)   →  5^6 = 15.625 estados possíveis
      - Tabela Q como defaultdict → vetores de zeros inicializados on-demand.
      - Política ε-greedy; ε controlado externamente via set_epsilon().
      - Método from_modelo() para reconstruir agente a partir de pickle.

    Estado : vetor de 6 floats em [0,1]  —  [d_0, d+30, d-30, d+60, d-60, v_norm]
    Ações  : 5  —  0=nada, 1=acelerar, 2=frear, 3=esq, 4=dir

    Update TD (off-policy):
        alvo   = r + γ · max_a' Q(s', a')   (se não terminou)
        alvo   = r                            (se terminou)
        Q(s,a) ← Q(s,a) + α · (alvo − Q(s,a))
    """

    def __init__(self, obs_dim=6, n_actions=5, K=5,
                 alpha=0.15, gamma=0.97,
                 eps_inicial=1.0, eps_final=0.05):
        self.obs_dim   = obs_dim
        self.n_actions = n_actions
        self.K         = K
        self.alpha     = alpha
        self.gamma     = gamma
        self.eps       = eps_inicial
        self.eps_final = eps_final

        # Tabela Q: chave discreta (tupla) → Q-valores (np.array de n_actions)
        self.Q: dict = defaultdict(lambda: np.zeros(self.n_actions))

    # ------------------------------------------------------------------
    # Discretização
    # ------------------------------------------------------------------
    def discretizar(self, obs) -> tuple:
        """
        Converte vetor de 6 floats → tupla de 6 ints  (K bins uniformes).

        Exemplo (K=5, obs=[0.35, 1.00, 0.30, 0.41, 0.18, 0.50]):
            → (1, 4, 1, 2, 0, 2)
        """
        return tuple(min(int(v * self.K), self.K - 1) for v in obs)

    # ------------------------------------------------------------------
    # Política ε-greedy
    # ------------------------------------------------------------------
    def escolher_acao(self, obs) -> int:
        """
        ε-greedy: com prob ε explora aleatoriamente, senão age gulosamente.
        """
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        chave = self.discretizar(obs)
        return int(np.argmax(self.Q[chave]))

    def escolher_acao_gulosa(self, obs) -> int:
        """Sempre gulosa (ε=0) — usada na avaliação."""
        chave = self.discretizar(obs)
        return int(np.argmax(self.Q[chave]))

    # ------------------------------------------------------------------
    # Update TD
    # ------------------------------------------------------------------
    def atualizar(self, obs, acao, reward, obs_prox, terminou: bool):
        """Aplica a regra de atualização do Q-Learning."""
        s      = self.discretizar(obs)
        s_prox = self.discretizar(obs_prox)

        if terminou:
            alvo = reward
        else:
            alvo = reward + self.gamma * np.max(self.Q[s_prox])

        self.Q[s][acao] += self.alpha * (alvo - self.Q[s][acao])

    # ------------------------------------------------------------------
    # Controle de ε
    # ------------------------------------------------------------------
    def set_epsilon(self, eps: float):
        self.eps = max(self.eps_final, float(eps))

    # ------------------------------------------------------------------
    # Serialização / desserialização
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "q_table":          dict(self.Q),
            "discretization_K": self.K,
            "obs_dim":          self.obs_dim,
            "n_actions":        self.n_actions,
            "alpha":            self.alpha,
            "gamma":            self.gamma,
        }

    @classmethod
    def from_modelo(cls, modelo: dict) -> "AgenteQLearning":
        """
        Reconstrói agente a partir do dicionário salvo no pickle.
        ε = 0 (modo avaliação gulosa).
        """
        cfg = modelo.get("config", {})
        agente = cls(
            obs_dim   = modelo.get("obs_dim", 6),
            n_actions = modelo.get("n_actions", 5),
            K         = modelo["discretization_K"],
            alpha     = cfg.get("alpha", 0.15),
            gamma     = cfg.get("gamma", 0.97),
            eps_inicial = 0.0,
            eps_final   = 0.0,
        )
        for k, v in modelo["q_table"].items():
            agente.Q[k] = np.array(v)
        agente.eps = 0.0
        return agente


# ============================================================================
# LOOP DE TREINAMENTO  (round-robin nas 16 pistas de treino)
# ============================================================================

def treinar_round_robin(pistas_treino, agente, n_episodios_por_pista,
                        max_passos, decaimento_eps_episodios, verbose=True):
    """
    Treina em round-robin: sorteia uma pista a cada episódio.

    Schedule ε: decaimento LINEAR de eps_inicial → eps_final ao longo de
    `decaimento_eps_episodios` episódios; depois mantém eps_final fixo.

    Retorna
    -------
    historico_recompensas : list[float]
    historico_sucessos    : list[bool]
    rewards_por_pista     : dict[str, list[float]]
    """
    historico_recompensas = []
    historico_sucessos    = []
    rewards_por_pista     = {p: [] for p in pistas_treino}

    n_total     = n_episodios_por_pista * len(pistas_treino)
    eps_inicial = agente.eps

    # Cache: não recalcula BFS a cada episódio
    envs = {
        p: AmbienteCarro(p, max_steps=max_passos, seed=SEED)
        for p in pistas_treino
    }

    janela = 500
    recompensas_janela: list = []

    for ep in range(n_total):

        # ── Schedule ε linear ─────────────────────────────────────────
        if decaimento_eps_episodios > 0:
            frac    = min(ep / decaimento_eps_episodios, 1.0)
            eps_novo = eps_inicial + frac * (agente.eps_final - eps_inicial)
        else:
            eps_novo = agente.eps_final
        agente.set_epsilon(eps_novo)

        # ── Sorteia pista ─────────────────────────────────────────────
        pista = random.choice(pistas_treino)
        env   = envs[pista]

        # ── Episódio ──────────────────────────────────────────────────
        obs          = env.reset()
        done         = False
        reward_total = 0.0
        sucesso      = False

        while not done:
            acao = agente.escolher_acao(obs)
            obs_prox, reward, term, trunc, info = env.step(acao)
            agente.atualizar(obs, acao, reward, obs_prox, term)
            obs           = obs_prox
            reward_total += reward
            done          = term or trunc
            if info.get("chegada"):
                sucesso = True

        # ── Registros ─────────────────────────────────────────────────
        historico_recompensas.append(reward_total)
        historico_sucessos.append(sucesso)
        rewards_por_pista[pista].append(reward_total)
        recompensas_janela.append(reward_total)
        if len(recompensas_janela) > janela:
            recompensas_janela.pop(0)

        # ── Log a cada 5000 episódios ──────────────────────────────────
        if verbose and (ep + 1) % 5_000 == 0:
            media  = np.mean(recompensas_janela)
            taxa   = np.mean(historico_sucessos[-janela:]) * 100
            estados = len(agente.Q)
            print(
                f"Ep {ep+1:>7}/{n_total}  "
                f"ε={agente.eps:.3f}  "
                f"R̄(ult.{janela})={media:>8.1f}  "
                f"sucesso={taxa:>5.1f}%  "
                f"estados={estados:>6}"
            )

    return historico_recompensas, historico_sucessos, rewards_por_pista


# ============================================================================
# AVALIAÇÃO  (ε = 0)
# ============================================================================

def avaliar(env, agente, n_episodios=10) -> dict:
    """
    Roda n_episodios com política totalmente gulosa (ε=0).

    Retorna
    -------
    n_passos         : passos do melhor episódio com chegada (ou do último)
    recompensa_total : recompensa do mesmo episódio
    sucesso          : chegou ao menos uma vez?
    velocidade_media : média de v ao longo do episódio representativo
    velocidade_maxima: maior v registrado no episódio representativo
    taxa_sucesso     : fração de episódios com chegada
    """
    eps_backup = agente.eps
    agente.eps = 0.0

    melhores = {
        "n_passos":          None,
        "recompensa_total":  None,
        "sucesso":           False,
        "velocidade_media":  0.0,
        "velocidade_maxima": 0.0,
    }
    sucessos = 0

    for _ in range(n_episodios):
        obs  = env.reset()
        done = False
        reward_ep   = 0.0
        passos      = 0
        velocidades = []

        while not done:
            acao = agente.escolher_acao_gulosa(obs)
            obs_prox, reward, term, trunc, info = env.step(acao)
            reward_ep  += reward
            passos     += 1
            velocidades.append(obs_prox[-1] * 2.0)   # v_norm * V_MAX (2.0 fixo do enunciado)
            done        = term or trunc
            obs         = obs_prox

        chegou = info.get("chegada", False)
        if chegou:
            sucessos += 1

        # Mantém o melhor episódio com chegada; se nunca chegou, guarda o último
        if chegou and not melhores["sucesso"]:
            melhores.update({
                "n_passos":          passos,
                "recompensa_total":  reward_ep,
                "sucesso":           True,
                "velocidade_media":  float(np.mean(velocidades)) if velocidades else 0.0,
                "velocidade_maxima": float(np.max(velocidades))  if velocidades else 0.0,
            })
        elif not melhores["sucesso"]:
            melhores.update({
                "n_passos":          passos,
                "recompensa_total":  reward_ep,
                "velocidade_media":  float(np.mean(velocidades)) if velocidades else 0.0,
                "velocidade_maxima": float(np.max(velocidades))  if velocidades else 0.0,
            })

    melhores["taxa_sucesso"] = sucessos / n_episodios
    agente.eps = eps_backup
    return melhores


# ============================================================================
# SALVAR / CARREGAR MODELO
# ============================================================================

def treinar_ou_carregar(nome, fn_treinar, recarregar=False):
    arquivo = DIR_TREINAMENTO / f"{nome}.pkl"
    if arquivo.exists() and not recarregar:
        print(f"Carregando {arquivo} ...")
        with open(arquivo, "rb") as f:
            return pickle.load(f)
    print(f"Treinando {nome} ...")
    resultado = fn_treinar()
    with open(arquivo, "wb") as f:
        pickle.dump(resultado, f)
    print(f"Salvo em {arquivo}")
    return resultado


# ============================================================================
# GERAÇÃO DOS ARQUIVOS DE SAÍDA  (formato README §4.3)
# ============================================================================

def escrever_saida(caminho, nome_algoritmo, pista,
                   resultado_avaliacao, n_episodios_treinados, n_estados):
    r           = resultado_avaliacao
    sucesso_str = "SIM" if r["sucesso"] else "NAO"
    n_passos    = r["n_passos"]          if r["n_passos"]         is not None else "N/A"
    recompensa  = f"{r['recompensa_total']:.2f}" if r["recompensa_total"] is not None else "N/A"
    v_media     = f"{r['velocidade_media']:.3f}"
    v_max       = f"{r['velocidade_maxima']:.3f}"

    conteudo = (
        f"=== Pista: {Path(pista).name} ===\n"
        f"Algoritmo: {nome_algoritmo}\n"
        f"Episódios totais de treinamento: {n_episodios_treinados}\n"
        f"Estados populados: {n_estados}\n"
        f"Tempo de chegada (passos): {n_passos}\n"
        f"Velocidade média: {v_media}\n"
        f"Velocidade máxima atingida: {v_max}\n"
        f"Recompensa total: {recompensa}\n"
        f"Sucesso: {sucesso_str}\n"
    )

    Path(caminho).write_text(conteudo, encoding="utf-8")
    print(f"\n  → {caminho}")
    print(conteudo)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodios-por-pista", type=int, default=30_000,
                        help="Episódios de treino por pista (default: 30000)")
    parser.add_argument("--max-passos", type=int, default=500)
    parser.add_argument("--K", type=int, default=5,
                        help="Baldes de discretização (default: 5)")
    parser.add_argument("--alpha", type=float, default=0.15,
                        help="Taxa de aprendizado α (default: 0.15)")
    parser.add_argument("--gamma", type=float, default=0.97,
                        help="Fator de desconto γ (default: 0.97)")
    parser.add_argument("--eps-inicial", type=float, default=1.0)
    parser.add_argument("--eps-final",   type=float, default=0.05)
    parser.add_argument("--recarregar", action="store_true",
                        help="Força re-treino mesmo se o pickle existir")
    parser.add_argument("--avaliar", type=str, default=None,
                        help="Apenas avalia o modelo salvo na pista especificada")
    args = parser.parse_args()

    # ─── Treinamento round-robin ─────────────────────────────────────────
    def fn_treinar():
        agente = AgenteQLearning(
            obs_dim=6, n_actions=5, K=args.K,
            alpha=args.alpha, gamma=args.gamma,
            eps_inicial=args.eps_inicial, eps_final=args.eps_final,
        )
        n_total    = args.episodios_por_pista * len(PISTAS_TREINO)
        decaimento = int(0.80 * n_total)   # decai em 80% dos episódios

        print(f"\n{'='*60}")
        print(f"  Q-LEARNING — round-robin em {len(PISTAS_TREINO)} pistas")
        print(f"  Episódios totais : {n_total:,}")
        print(f"  α={agente.alpha}  γ={agente.gamma}  K={agente.K}")
        print(f"  ε: {agente.eps:.2f} → {agente.eps_final:.2f}  (decay em {decaimento:,} ep)")
        print(f"{'='*60}\n")

        rewards, sucessos, rewards_por_pista = treinar_round_robin(
            PISTAS_TREINO, agente,
            n_episodios_por_pista=args.episodios_por_pista,
            max_passos=args.max_passos,
            decaimento_eps_episodios=decaimento,
        )

        # Resumo final por pista
        print("\n── Resumo por pista (últimos 500 ep) ──")
        for p in PISTAS_TREINO:
            ult   = rewards_por_pista[p][-500:] if rewards_por_pista[p] else []
            media = np.mean(ult) if ult else float("nan")
            print(f"  {Path(p).name}  R̄={media:>8.1f}")

        return {
            "q_table":            dict(agente.Q),
            "discretization_K":   args.K,
            "obs_dim":            6,
            "n_actions":          5,
            "n_episodes_trained": n_total,
            "rewards_history":    rewards,
            "rewards_por_pista":  rewards_por_pista,
            "config": {
                "alpha":       agente.alpha,
                "gamma":       agente.gamma,
                "eps_inicial": args.eps_inicial,
                "eps_final":   args.eps_final,
            },
            "seed":        SEED,
            "tracks_used": PISTAS_TREINO,
        }

    if args.avaliar:
        arquivo_pkl = DIR_TREINAMENTO / "qlearning.pkl"
        if not arquivo_pkl.exists():
            print(f"ERRO: {arquivo_pkl} não encontrado. Treine o modelo primeiro.")
            sys.exit(1)
        print(f"Carregando {arquivo_pkl} ...")
        with open(arquivo_pkl, "rb") as f:
            modelo = pickle.load(f)
    else:
        modelo = treinar_ou_carregar("qlearning", fn_treinar,
                                     recarregar=args.recarregar)

    # ─── Reconstrói agente para avaliação ────────────────────────────────
    agente_aval = AgenteQLearning.from_modelo(modelo)
    n_estados   = len(agente_aval.Q)
    n_ep_treino = modelo["n_episodes_trained"]
    nome_alg    = "Q-Learning (round-robin em pistas 01-16)"

    print(f"\nEstados na tabela Q : {n_estados:,}")
    print(f"Episódios treinados : {n_ep_treino:,}")

    # ─── Avaliação ────────────────────────────────────────────────────────
    pistas_avaliar = [args.avaliar] if args.avaliar else PISTAS_HOLDOUT
    print(f"\n{'='*60}")
    print("  AVALIAÇÃO (política gulosa, ε=0)")
    print(f"{'='*60}")

    for pista in pistas_avaliar:
        print(f"\nAvaliando {pista} ...")
        env       = AmbienteCarro(pista, max_steps=args.max_passos, seed=SEED)
        resultado = avaliar(env, agente_aval, n_episodios=10)

        nome_pista = Path(pista).stem            # "pista_17"
        arq_saida  = f"q_learning_{nome_pista}.txt"
        escrever_saida(
            arq_saida, nome_alg, pista,
            resultado, n_ep_treino, n_estados,
        )

    print("\nPronto.")


if __name__ == "__main__":
    main()
