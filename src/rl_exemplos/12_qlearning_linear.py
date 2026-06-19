"""
12_qlearning_linear.py — Seção 7: Q-learning com aproximação linear
Q(s,a) = w · f(s,a)  com features de estado-ação em blocos.
Treina no Mundo Grade, exibe os pesos aprendidos e a política.
Compara diretamente com a versão tabular.
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

# ── Features de estado (4 features, normalizadas) ─────────────
NOMES_F = ['dist_obj', 'dist_neg', 'col_norm', 'lin_norm']
N_F = 4

def feat_estado(s):
    return np.array([
        (abs(s[0]-T_POS[0]) + abs(s[1]-T_POS[1])) / D_MAX,   # dist. ao +1
        (abs(s[0]-T_NEG[0]) + abs(s[1]-T_NEG[1])) / D_MAX,   # dist. ao -1
        (s[0] - 1) / 3.0,                                       # coluna normalizada
        (s[1] - 1) / 2.0,                                       # linha normalizada
    ])

def phi(s, a):
    """Vetor de estado-ação: N_F × |ACOES| — só o bloco da ação a é não-zero."""
    idx = ACOES.index(a)
    v   = np.zeros(N_F * len(ACOES))
    v[idx * N_F : (idx + 1) * N_F] = feat_estado(s)
    return v

def Q_aprox(s, a, w):
    return float(w @ phi(s, a))

# ── Q-learning com aproximação linear ────────────────────────
def treinar_linear(n_ep=6000, alpha=0.02, gamma=0.9, eps=0.25, seed=42):
    random.seed(seed)
    w = np.zeros(N_F * len(ACOES))
    retornos = []
    for _ in range(n_ep):
        s = random.choice(NAO_T)
        G = 0.0
        for _ in range(120):
            if s in (T_POS, T_NEG): break
            a  = (random.choice(ACOES) if random.random() < eps
                  else max(ACOES, key=lambda x: Q_aprox(s, x, w)))
            s2 = amostrar(s, a)
            r  = recomp(s2)
            G += r
            max_q2 = max(Q_aprox(s2, x, w) for x in ACOES)
            delta  = r + gamma * max_q2 - Q_aprox(s, a, w)
            w     += alpha * delta * phi(s, a)       # ← regra central
            s = s2
        retornos.append(G)
    return w, retornos

# ── Q-learning tabular (referência) ──────────────────────────
def treinar_tabular(n_ep=6000, alpha=0.3, gamma=0.9, eps=0.2, seed=42):
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
            G += r
            Q[(s, a)] += alpha * (r + gamma * max(Q[(s2, x)] for x in ACOES) - Q[(s, a)])
            s = s2
        retornos.append(G)
    return Q, retornos

w_final, ret_lin = treinar_linear()
Q_tab,   ret_tab = treinar_tabular()

politica_lin = {s: max(ACOES, key=lambda a: Q_aprox(s, a, w_final)) for s in NAO_T}
politica_tab = {s: max(ACOES, key=lambda a: Q_tab[(s, a)])           for s in NAO_T}
SETA = {'N': '↑', 'S': '↓', 'L': '→', 'O': '←'}

# ── Impressão ─────────────────────────────────────────────────
print("=" * 60)
print("  Q-learning com Aproximação Linear — Mundo Grade")
print("=" * 60)
print("\n  Pesos aprendidos  w  (por bloco de ação):")
print(f"  {'Ação':>4}   " + "  ".join(f"{n:>10}" for n in NOMES_F))
print("  " + "─" * 52)
for i, a in enumerate(ACOES):
    bloco = w_final[i * N_F:(i + 1) * N_F]
    print(f"  {a:>4}   " + "  ".join(f"{v:>+10.4f}" for v in bloco))

print("\n  Comparação de políticas (Tab = tabular, Lin = linear):")
for row in range(3, 0, -1):
    lin = "  lin%d  " % row
    for col in range(1, 5):
        s = (col, row)
        if   s == (2, 2): lin += "  ██  "
        elif s == T_POS:  lin += " [+1] "
        elif s == T_NEG:  lin += " [-1] "
        else:             lin += f" {SETA[politica_lin[s]]}/{SETA[politica_tab[s]]} "
    print(lin)
print("  (Lin/Tab — divergências indicam onde a aproximação linear difere)")

# ── Gráfico ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Curvas de aprendizado
ax = axes[0]
ww = 100
for ret, cor, lbl in [(ret_lin,'b','Linear'), (ret_tab,'r','Tabular')]:
    ma = np.convolve(ret, np.ones(ww)/ww, mode='valid')
    ax.plot(ma, '-', color=cor, lw=2, label=lbl)
ax.set_xlabel('Episódio'); ax.set_ylabel(f'Retorno (média {ww} ep.)')
ax.set_title('Curvas de aprendizado'); ax.legend(); ax.grid(True, alpha=0.3)

# Pesos finais por bloco
ax2 = axes[1]
x = np.arange(N_F)
w_bp = 0.18
CORES = ['#2980b9', '#e74c3c', '#27ae60', '#f39c12']
for i, (a, c) in enumerate(zip(ACOES, CORES)):
    ax2.bar(x + i * w_bp, w_final[i*N_F:(i+1)*N_F],
            w_bp, label=a, color=c, alpha=0.85)
ax2.set_xticks(x + 1.5*w_bp); ax2.set_xticklabels(NOMES_F, fontsize=8)
ax2.axhline(0, color='k', lw=0.8)
ax2.set_title('Pesos aprendidos por ação'); ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis='y')

# Grade: política linear
ax3 = axes[2]
ax3.set_xlim(0.5, 4.5); ax3.set_ylim(0.5, 3.5)
ax3.set_aspect('equal'); ax3.axis('off')
ax3.set_title('Política aprendida (linear)')
for col in range(1, 5):
    for row in range(1, 4):
        s = (col, row)
        fc = ('#444' if s==(2,2) else '#90EE90' if s==T_POS
              else '#FFB6C1' if s==T_NEG else 'white')
        ax3.add_patch(plt.Rectangle((col-.5,row-.5),1,1,ec='gray',fc=fc,lw=1.2))
        if   s == T_POS: ax3.text(col,row,'+1',ha='center',va='center',
                                  fontsize=11,fontweight='bold')
        elif s == T_NEG: ax3.text(col,row,'−1',ha='center',va='center',
                                  fontsize=11,fontweight='bold')
        elif s != (2,2): ax3.text(col,row,SETA[politica_lin[s]],
                                  ha='center',va='center',fontsize=22)

plt.tight_layout()
plt.savefig('/home/claude/ar_exemplos/12_qlearning_linear.png',
            dpi=130, bbox_inches='tight')
plt.show()
