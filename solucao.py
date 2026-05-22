"""
Esqueleto da sua solução para o EP do carrinho (versão tabular).

Você deve implementar:
    - AgenteQLearning  (tabular)

E preencher main() para orquestrar treinamento, avaliação e geração dos arquivos
de saída descritos no README (q_learning.txt).

Uso:
    python solucao.py pistas/pista_03.txt

Termos como `step`, `reset`, `obs`, `action`, `reward` são mantidos em inglês
por serem o vocabulário canônico de Aprendizado por Reforço (Sutton & Barto).
"""

import sys
import random
import argparse
import pickle
from pathlib import Path

import numpy as np

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from env import AmbienteCarro  # noqa: E402
# from visualize import renderizar_episodio  # use isto para animar seu agente no terminal


# === Configuração ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Diretório onde os modelos treinados serão salvos via pickle (ver docs/anexo_b_pickle.md)
DIR_TREINAMENTO = Path("treinamento")
DIR_TREINAMENTO.mkdir(exist_ok=True)


# ============================================================================
# Q-LEARNING TABULAR
# ============================================================================

class AgenteQLearning:
    """
    TODO: implementar Q-Learning tabular com discretização do estado.

    Dicas:
    - O estado é um vetor de 6 floats em [0, 1]. Discretize cada componente
      em K baldes (sugerido K=5 para baseline) para chegar a uma chave discreta.
    - Use um dict {chave_discreta: np.array(n_actions)} ou um np.ndarray
      multidimensional para a tabela Q.
    - Atualização: Q(s,a) ← Q(s,a) + α [r + γ max_{a'} Q(s', a') − Q(s,a)]
    """

    def __init__(self, obs_dim, n_actions, K=5, alpha=0.1, gamma=0.99,
                 eps_inicial=1.0, eps_final=0.05):
        self.n_actions = n_actions
        self.K = K
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_inicial
        self.eps_final = eps_final
        # TODO: estrutura de dados para Q
        raise NotImplementedError

    def discretizar(self, obs):
        """Converte vetor float em chave discreta (tupla de ints)."""
        # TODO
        raise NotImplementedError

    def escolher_acao(self, obs):
        """Política ε-greedy."""
        # TODO
        raise NotImplementedError

    def atualizar(self, s, a, r, s_prox, terminou):
        """Aplica a regra de update do Q-Learning."""
        # TODO
        raise NotImplementedError


# ============================================================================
# LOOP DE TREINAMENTO
# ============================================================================

def treinar(env, agente, n_episodios, decaimento_eps_episodios, verbose=True):
    """
    Loop de treinamento. O agente decide o que fazer em cada passo via
    escolher_acao() e atualizar().
    """
    historico_recompensas = []
    historico_sucessos = []

    for ep in range(n_episodios):
        # TODO: schedule do epsilon (linear de eps_inicial a eps_final em
        #       decaimento_eps_episodios episódios).
        # TODO: loop do episódio:
        #   obs = env.reset()
        #   while not done:
        #       action = agente.escolher_acao(obs)
        #       obs_prox, reward, term, trunc, info = env.step(action)
        #       agente.atualizar(obs, action, reward, obs_prox, term)
        #       obs = obs_prox
        # TODO: registrar reward total e flag de sucesso (info.get("chegada"))
        pass

    return historico_recompensas, historico_sucessos


# ============================================================================
# AVALIAÇÃO (com ε = 0)
# ============================================================================

def avaliar(env, agente, n_episodios=10):
    """
    Roda n_episodios com política gulosa (ε = 0) e retorna estatísticas.

    Retorna: dict com {n_passos, recompensa_total, sucesso, velocidade_media}
    """
    # TODO
    raise NotImplementedError


# ============================================================================
# SALVAR / CARREGAR MODELO (ver docs/anexo_b_pickle.md)
# ============================================================================

def treinar_ou_carregar(nome, fn_treinar, recarregar=False):
    """
    Se 'treinamento/{nome}.pkl' existe e recarregar=False, carrega.
    Caso contrário, chama fn_treinar() e salva o resultado.
    """
    arquivo = DIR_TREINAMENTO / f"{nome}.pkl"
    if arquivo.exists() and not recarregar:
        print(f"Carregando {arquivo} ...")
        with open(arquivo, "rb") as f:
            return pickle.load(f)
    else:
        print(f"Treinando {nome} ...")
        resultado = fn_treinar()
        with open(arquivo, "wb") as f:
            pickle.dump(resultado, f)
        print(f"Salvo em {arquivo}")
        return resultado


# ============================================================================
# GERAÇÃO DOS ARQUIVOS DE SAÍDA
# ============================================================================

def escrever_saida(caminho, nome_algoritmo, env, resultado_avaliacao, n_episodios_treinados, K):
    """
    Escreve um arquivo no formato esperado pelo README (§3.3):

    === Pista: <nome> ===
    Algoritmo: <nome>
    Episódios de treinamento: N
    Discretização: K=<k>
    Tempo de chegada (passos): N
    Velocidade média: V
    Recompensa total: R
    Sucesso: SIM/NAO
    """
    # TODO: implementar conforme formato do README §3.3
    raise NotImplementedError


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pista", help="Caminho para arquivo de pista (treino principal)")
    parser.add_argument("--episodios", type=int, default=30_000)
    parser.add_argument("--max-passos", type=int, default=500)
    parser.add_argument("--K", type=int, default=5, help="Baldes da discretização (fixo em 5; ver README §2.2)")
    args = parser.parse_args()

    print(f"Carregando pista: {args.pista}")
    env = AmbienteCarro(args.pista, max_steps=args.max_passos, seed=SEED)
    print(f"  obs_dim = {env.obs_dim}, n_actions = {env.n_actions}")

    # ─── Q-Learning Tabular ────────────────────────────────────────────────
    print("\n=== Q-Learning Tabular ===")
    # agente = AgenteQLearning(env.obs_dim, env.n_actions, K=args.K)
    # rewards, sucessos = treinar(env, agente, args.episodios,
    #                             decaimento_eps_episodios=int(0.8 * args.episodios))
    # resultado = avaliar(env, agente)
    # escrever_saida("q_learning.txt", "Q-Learning", env, resultado,
    #                args.episodios, args.K)

    print("\nPronto.")


if __name__ == "__main__":
    main()
