"""
08_sarsa_vs_qlearning.py — Seção 6: SARSA vs Q-learning
Ambiente Cliff Walking (Sutton & Barto):
  grade 4×12, início (0,0), objetivo (11,0),
  penhasco: (1,0)–(10,0) → penalidade −100, volta ao início.
Mostra que Q-learning aprende o caminho rente ao penhasco (off-policy)
enquanto SARSA aprende o caminho mais seguro (on-policy).
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ── Ambiente Cliff Walking ────────────────────────────────────
COLS, ROWS = 12, 4
INICIO     = (0, 0)
GOAL       = (11, 0)
CLIFF      = {(c, 0) for c in range(1, 11)}
ACOES      = ['N', 'S', 'L', 'O']
DELTA      = {'N': (0, 1), 'S': (0, -1), 'L': (1, 0), 'O': (-1, 0)}

def mover_cliff(s, a):
    c, r = s; dc, dr = DELTA[a]
    s2 = (c + dc, r + dr)
    if 0 <= s2[0] < COLS and 0 <= s2[1] < ROWS:
        return s2
    return s

def passo(s, a):
    """Retorna (s', r) com lógica do penhasco."""
    s2 = mover_cliff(s, a)
    if s2 in CLIFF:
        return INICIO, -100.0   # cai: penalidade severa e volta ao início
    return s2, -1.0             # passo normal

def eps_greedy(Q, s, eps):
    return (random.choice(ACOES) if random.random() < eps
            else max(ACOES, key=lambda a: Q[(s, a)]))

# ── Q-learning (off-policy) ───────────────────────────────────
def treinar_ql(n_ep=500, alpha=0.5, gamma=1.0, eps=0.1, seed=42):
    random.seed(seed)
    Q = defaultdict(float)
    retornos = []
    for _ in range(n_ep):
        s, G = INICIO, 0
        for _ in range(500):
            if s == GOAL: break
            a = eps_greedy(Q, s, eps)
            s2, r = passo(s, a)
            G += r
            max_q2 = max(Q[(s2, x)] for x in ACOES)
            Q[(s, a)] += alpha * (r + gamma * max_q2 - Q[(s, a)])
            s = s2
        retornos.append(max(G, -500))
    return Q, retornos

# ── SARSA (on-policy) ─────────────────────────────────────────
def treinar_sarsa(n_ep=500, alpha=0.5, gamma=1.0, eps=0.1, seed=42):
    random.seed(seed)
    Q = defaultdict(float)
    retornos = []
    for _ in range(n_ep):
        s = INICIO
        a = eps_greedy(Q, s, eps)
        G = 0
        for _ in range(500):
            if s == GOAL: break
            s2, r = passo(s, a)
            G += r
            a2 = eps_greedy(Q, s2, eps)
            Q[(s, a)] += alpha * (r + gamma * Q[(s2, a2)] - Q[(s, a)])
            s, a = s2, a2
        retornos.append(max(G, -500))
    return Q, retornos

Q_ql,    ret_ql    = treinar_ql()
Q_sarsa, ret_sarsa = treinar_sarsa()

# ── Trajetória gulosa ─────────────────────────────────────────
def traj_gulosa(Q, max_p=100):
    s, traj = INICIO, [INICIO]
    for _ in range(max_p):
        if s == GOAL: break
        a = max(ACOES, key=lambda a: Q[(s, a)])
        s2, _ = passo(s, a)
        if s2 in CLIFF: s2 = INICIO
        traj.append(s2); s = s2
    return traj

tj_ql    = traj_gulosa(Q_ql)
tj_sarsa = traj_gulosa(Q_sarsa)

print("=" * 55)
print("  SARSA vs Q-learning — Cliff Walking")
print("=" * 55)
print(f"\n  Trajetória Q-learning (gulosa): {len(tj_ql)} passos")
print(f"  Trajetória SARSA     (gulosa): {len(tj_sarsa)} passos")
w = 100
print(f"\n  Retorno médio (últimos {w} episódios):")
print(f"    Q-learning: {np.mean(ret_ql[-w:]):.1f}")
print(f"    SARSA:      {np.mean(ret_sarsa[-w:]):.1f}")
print("\n  Interpretação:")
print("    Q-learning aprende o caminho ótimo para um agente SEM exploração.")
print("    SARSA aprende o caminho mais seguro dado que EXPLORA com ε=0.1.")

# ── Gráfico ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Curvas de aprendizado
ax = axes[0]
ww = 20
ax.plot(np.convolve(ret_ql,    np.ones(ww)/ww, mode='valid'),
        'b-', lw=2, label='Q-learning (off-policy)')
ax.plot(np.convolve(ret_sarsa, np.ones(ww)/ww, mode='valid'),
        'r-', lw=2, label='SARSA (on-policy)')
ax.set_xlabel('Episódio'); ax.set_ylabel(f'Retorno (média móvel, w={ww})')
ax.set_title('Curvas de aprendizado'); ax.set_ylim(-150, 5)
ax.legend(); ax.grid(True, alpha=0.3)

# Trajetórias sobre a grade
ax2 = axes[1]
for row in range(ROWS):
    for col in range(COLS):
        s = (col, row)
        fc = ('#FFB6C1' if s in CLIFF else
              '#ADD8E6' if s == INICIO else
              '#90EE90' if s == GOAL  else 'white')
        ax2.add_patch(plt.Rectangle((col - .5, row - .5), 1, 1,
                                    ec='#aaa', fc=fc, lw=0.5))
ax2.text(INICIO[0], INICIO[1], 'S', ha='center', va='center',
         fontsize=9, fontweight='bold')
ax2.text(GOAL[0],   GOAL[1],   'G', ha='center', va='center',
         fontsize=9, fontweight='bold')
ax2.text(5.5, -.35, '← penhasco →', ha='center', fontsize=8, color='#c0392b')

for traj, cor, lbl, dy in [(tj_ql, 'blue', 'Q-learning', +0.10),
                            (tj_sarsa, 'red', 'SARSA', -0.10)]:
    xs = [s[0] for s in traj]
    ys = [s[1] + dy for s in traj]
    ax2.plot(xs, ys, '-o', color=cor, ms=3, lw=2, label=lbl)

ax2.set_xlim(-0.6, 11.6); ax2.set_ylim(-0.7, 3.6)
ax2.set_aspect('equal'); ax2.axis('off')
ax2.set_title('Trajetórias gulosas aprendidas')
ax2.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('/home/claude/ar_exemplos/08_sarsa_vs_qlearning.png',
            dpi=130, bbox_inches='tight')
plt.show()
