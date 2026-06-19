"""
06_qlearning_tabular.py — Seção 5: Q-learning tabular
Mundo Grade 4×3 com transições estocásticas.
Treina Q-learning e exibe:
  • Curva de aprendizado (retorno por episódio)
  • Tabela Q final (linhas = estados, colunas = ações)
  • Política ótima derivada com setas sobre a grade
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ── Mundo Grade 4×3 ──────────────────────────────────────────
ESTADOS  = sorted({(c, r) for c in range(1, 5)
                   for r in range(1, 4)} - {(2, 2)})
NAO_T    = [s for s in ESTADOS if s not in ((4, 3), (4, 2))]
T_POS, T_NEG = (4, 3), (4, 2)
ACOES    = ['N', 'S', 'L', 'O']
DELTA    = {'N': (0, 1), 'S': (0, -1), 'L': (1, 0), 'O': (-1, 0)}
PERP     = {'N': ('O', 'L'), 'S': ('L', 'O'), 'L': ('N', 'S'), 'O': ('S', 'N')}

def mover(s, a):
    c, r = s; dc, dr = DELTA[a]
    s2 = (c + dc, r + dr)
    return s2 if s2 in ESTADOS else s

def amostrar(s, a):
    ae, ad = PERP[a]
    return random.choices(
        [mover(s, a), mover(s, ae), mover(s, ad)],
        weights=[0.8, 0.1, 0.1])[0]

def recomp(s):
    return {T_POS: +1.0, T_NEG: -1.0}.get(s, -0.04)

# ── Q-learning ────────────────────────────────────────────────
def qlearning(n_ep=3000, alpha=0.3, gamma=0.9, eps=0.2, seed=0):
    random.seed(seed)
    Q = defaultdict(float)
    retornos = []
    for _ in range(n_ep):
        s = random.choice(NAO_T)
        G = 0.0
        for _ in range(150):
            if s in (T_POS, T_NEG): break
            a = (random.choice(ACOES) if random.random() < eps
                 else max(ACOES, key=lambda x: Q[(s, x)]))
            s2 = amostrar(s, a)
            r  = recomp(s2)
            G += r
            max_q2 = max(Q[(s2, x)] for x in ACOES)
            delta  = r + gamma * max_q2 - Q[(s, a)]
            Q[(s, a)] += alpha * delta
            s = s2
        retornos.append(G)
    return Q, retornos

Q, retornos = qlearning()

# ── Derivar política ──────────────────────────────────────────
SETA     = {'N': '↑', 'S': '↓', 'L': '→', 'O': '←'}
politica = {s: max(ACOES, key=lambda a: Q[(s, a)]) for s in NAO_T}

print("=" * 56)
print("  Q-learning Tabular — Mundo Grade 4×3")
print("=" * 56)
print("\n  Política aprendida:")
for row in range(3, 0, -1):
    linha = f"  lin{row}  "
    for col in range(1, 5):
        s = (col, row)
        if   s == (2, 2):  linha += "  ██  "
        elif s == T_POS:   linha += " [+1] "
        elif s == T_NEG:   linha += " [-1] "
        else:              linha += f"  {SETA[politica[s]]}   "
    print(linha)

print("\n  Tabela Q (amostra de estados, melhor ação destacada):")
print(f"  {'Estado':>8}  {'N':>7}  {'S':>7}  {'L':>7}  {'O':>7}  Melhor")
print("  " + "─" * 52)
for s in [(1,1),(1,2),(1,3),(3,1),(3,2),(3,3)]:
    vals = {a: Q[(s, a)] for a in ACOES}
    mel  = max(ACOES, key=lambda a: vals[a])
    print(f"  {str(s):>8}  " +
          "  ".join(f"{vals[a]:>7.4f}" for a in ACOES) +
          f"  {mel}")

# ── Gráfico ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Curva de aprendizado
ax = axes[0]
w = 80
ma = np.convolve(retornos, np.ones(w) / w, mode='valid')
ax.plot(retornos, alpha=0.15, color='steelblue', lw=0.5)
ax.plot(range(w - 1, len(retornos)), ma, color='blue', lw=2,
        label=f'média móvel ({w} ep.)')
ax.set_xlabel('Episódio'); ax.set_ylabel('Retorno acumulado')
ax.set_title('Curva de aprendizado — Q-learning')
ax.legend(); ax.grid(True, alpha=0.3)

# Grade com política
ax2 = axes[1]
ax2.set_xlim(0.5, 4.5); ax2.set_ylim(0.5, 3.5)
ax2.set_aspect('equal'); ax2.axis('off')
ax2.set_title('Política aprendida pelo Q-learning')
for col in range(1, 5):
    for row in range(1, 4):
        s = (col, row)
        fc = ('#444'    if s == (2, 2)  else
              '#90EE90' if s == T_POS   else
              '#FFB6C1' if s == T_NEG   else 'white')
        ax2.add_patch(plt.Rectangle((col - .5, row - .5), 1, 1,
                                    ec='gray', fc=fc, lw=1.2))
        if   s == T_POS:  ax2.text(col, row, '+1', ha='center', va='center',
                                   fontsize=12, fontweight='bold')
        elif s == T_NEG:  ax2.text(col, row, '−1', ha='center', va='center',
                                   fontsize=12, fontweight='bold')
        elif s != (2, 2): ax2.text(col, row, SETA[politica[s]],
                                   ha='center', va='center', fontsize=22)

plt.tight_layout()
plt.savefig('/home/claude/ar_exemplos/06_qlearning_tabular.png',
            dpi=130, bbox_inches='tight')
plt.show()
