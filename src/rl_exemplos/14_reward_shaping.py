"""
14_reward_shaping.py — Seção 8: Reward Shaping
Compara Q-learning no Mundo Grade em três cenários:
  1. Sem shaping
  2. Shaping bom:  Φ(s) ∝ proximidade ao terminal +1
  3. Shaping ruim: Φ(s) ∝ proximidade ao terminal −1  (enganoso)
Mostra o efeito na velocidade de convergência e inclui exemplo numérico.
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
D_MAX    = 6.0

def mover(s, a):
    c, r = s; dc, dr = DELTA[a]
    s2 = (c + dc, r + dr)
    return s2 if s2 in ESTADOS else s

def amostrar(s, a):
    ae, ad = PERP[a]
    return random.choices([mover(s,a), mover(s,ae), mover(s,ad)],
                          weights=[0.8, 0.1, 0.1])[0]

def recomp(s):
    return {T_POS: +1.0, T_NEG: -1.0}.get(s, -0.04)

# ── Funções potencial ─────────────────────────────────────────
def phi_bom(s):
    """Φ(s) = 1 − dist(s, terminal+) / D_MAX  [↑ perto do objetivo]"""
    d = abs(s[0]-T_POS[0]) + abs(s[1]-T_POS[1])
    return 1.0 - d / D_MAX

def phi_ruim(s):
    """Φ(s) = 1 − dist(s, terminal−) / D_MAX  [↑ perto do perigo — enganoso!]"""
    d = abs(s[0]-T_NEG[0]) + abs(s[1]-T_NEG[1])
    return 1.0 - d / D_MAX

def shaping(s, s2, gamma, phi_fn):
    """F(s,s') = γ·Φ(s') − Φ(s)"""
    return gamma * phi_fn(s2) - phi_fn(s)

# ── Exemplo numérico (reproduz o texto) ──────────────────────
GAMMA = 0.9
s_ex, s2_bom, s2_ruim = (1, 1), (1, 2), (2, 1)

print("=" * 60)
print("  Reward Shaping baseado em potencial — Mundo Grade")
print("=" * 60)
print()
print("  ─── Exemplo numérico ────────────────────────────────────")
print(f"  Estado atual s = {s_ex}  →  próximo estado s' = {s2_bom}")
print(f"\n  Φ_bom({s_ex})  = {phi_bom(s_ex):.4f}")
print(f"  Φ_bom({s2_bom}) = {phi_bom(s2_bom):.4f}")
F_bom = shaping(s_ex, s2_bom, GAMMA, phi_bom)
print(f"  F_bom = γ·Φ({s2_bom}) − Φ({s_ex})"
      f" = {GAMMA}×{phi_bom(s2_bom):.4f} − {phi_bom(s_ex):.4f} = {F_bom:+.4f}")
print(f"  → Transição recebe bônus positivo: agente se aproximou do objetivo.\n")

print(f"  Φ_ruim({s_ex})  = {phi_ruim(s_ex):.4f}")
print(f"  Φ_ruim({s2_bom}) = {phi_ruim(s2_bom):.4f}")
F_ruim = shaping(s_ex, s2_bom, GAMMA, phi_ruim)
print(f"  F_ruim = γ·Φ({s2_bom}) − Φ({s_ex})"
      f" = {GAMMA}×{phi_ruim(s2_bom):.4f} − {phi_ruim(s_ex):.4f} = {F_ruim:+.4f}")
print(f"  → Shaping ruim: penaliza por se afastar do perigo (enganoso!).\n")

# ── Treinamento comparativo ────────────────────────────────────
def treinar(n_ep=2500, alpha=0.3, gamma=0.9, eps=0.3,
            phi_fn=None, seed=0):
    random.seed(seed)
    Q = defaultdict(float)
    retornos = []
    for _ in range(n_ep):
        s = random.choice(NAO_T)
        G = 0.0
        for _ in range(120):
            if s in (T_POS, T_NEG): break
            a  = (random.choice(ACOES) if random.random() < eps
                  else max(ACOES, key=lambda x: Q[(s, x)]))
            s2 = amostrar(s, a)
            r  = recomp(s2)
            F  = shaping(s, s2, gamma, phi_fn) if phi_fn else 0.0
            G += r
            max_q2 = max(Q[(s2, x)] for x in ACOES)
            Q[(s, a)] += alpha * (r + F + gamma * max_q2 - Q[(s, a)])
            s = s2
        retornos.append(G)
    return Q, retornos

Q_sem,  ret_sem  = treinar(phi_fn=None,     seed=1)
Q_bom,  ret_bom  = treinar(phi_fn=phi_bom,  seed=1)
Q_ruim, ret_ruim = treinar(phi_fn=phi_ruim, seed=1)

w = 100
print("  ─── Retorno médio (últimos 100 episódios) ───────────────")
print(f"  Sem shaping:   {np.mean(ret_sem[-w:]):.4f}")
print(f"  Shaping bom:   {np.mean(ret_bom[-w:]):.4f}")
print(f"  Shaping ruim:  {np.mean(ret_ruim[-w:]):.4f}")

# ── Gráfico ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ww = 60
ax = axes[0]
for ret, cor, lbl in [(ret_sem, 'steelblue', 'Sem shaping'),
                      (ret_bom, 'green',      'Shaping bom  (Φ → +1)'),
                      (ret_ruim,'red',         'Shaping ruim (Φ → −1)')]:
    ma = np.convolve(ret, np.ones(ww)/ww, mode='valid')
    ax.plot(ma, '-', color=cor, lw=2, label=lbl)
ax.set_xlabel('Episódio'); ax.set_ylabel(f'Retorno (média {ww} ep.)')
ax.set_title('Efeito do reward shaping na convergência')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Mapa de Φ_bom sobre a grade
ax2 = axes[1]
ax2.set_xlim(0.5, 4.5); ax2.set_ylim(0.5, 3.5)
ax2.set_aspect('equal'); ax2.axis('off')
ax2.set_title('Φ_bom(s) — potencial baseado em dist. ao objetivo +1')

phi_vals = {s: phi_bom(s) for s in ESTADOS}
vmin, vmax = min(phi_vals.values()), max(phi_vals.values())
cmap = plt.cm.YlGn

for col in range(1, 5):
    for row in range(1, 4):
        s = (col, row)
        if s == (2, 2):
            fc = '#333'
        elif s == T_POS:
            fc = '#27ae60'
        elif s == T_NEG:
            fc = '#e74c3c'
        else:
            t = (phi_vals[s] - vmin) / (vmax - vmin + 1e-9)
            fc = cmap(0.2 + 0.7 * t)
        ax2.add_patch(plt.Rectangle((col-.5, row-.5), 1, 1, ec='gray', fc=fc, lw=1.2))
        if s == T_POS:
            ax2.text(col, row, '+1', ha='center', va='center',
                     fontsize=11, fontweight='bold', color='white')
        elif s == T_NEG:
            ax2.text(col, row, '−1', ha='center', va='center',
                     fontsize=11, fontweight='bold', color='white')
        elif s != (2, 2):
            ax2.text(col, row, f'Φ={phi_vals[s]:.2f}',
                     ha='center', va='center', fontsize=8)

sm = plt.cm.ScalarMappable(cmap=cmap,
     norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
plt.colorbar(sm, ax=ax2, shrink=0.7, label='Φ_bom(s)')

plt.suptitle('Reward Shaping baseado em potencial: Φ(s) = 1 − dist_norm(s, objetivo)',
             fontsize=10)
plt.tight_layout()
plt.savefig('/home/claude/ar_exemplos/14_reward_shaping.png',
            dpi=130, bbox_inches='tight')
plt.show()
