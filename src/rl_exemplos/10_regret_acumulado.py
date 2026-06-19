"""
10_regret_acumulado.py — Seção 6: Regret acumulado
Compara ε-greedy (ε fixo) com a função de exploração f(u,n) = u + k/n
no Mundo Grade. O regret conta passos em que a ação escolhida
não foi a melhor ação disponível segundo Q atual.
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
    return random.choices([mover(s,a), mover(s,ae), mover(s,ad)],
                          weights=[0.8, 0.1, 0.1])[0]

def recomp(s):
    return {T_POS: +1.0, T_NEG: -1.0}.get(s, -0.04)

# ── Treinamento com medição de regret ─────────────────────────
def treinar(estrategia, n_passos=12000,
            alpha=0.3, gamma=0.9, eps=0.1, k=1.0, seed=0):
    random.seed(seed)
    Q  = defaultdict(float)
    N  = defaultdict(int)
    regret_serie = []
    regret_acum  = 0
    passo = 0

    while passo < n_passos:
        s = random.choice(NAO_T)
        for _ in range(40):
            if s in (T_POS, T_NEG) or passo >= n_passos:
                break
            # Escolha de ação segundo a estratégia
            if estrategia == 'eps_greedy':
                a = (random.choice(ACOES) if random.random() < eps
                     else max(ACOES, key=lambda x: Q[(s, x)]))
            else:   # funcao_exp
                a = max(ACOES, key=lambda x: Q[(s, x)] + k / (N[(s, x)] + 1))

            # Regret: ação não foi a melhor disponível?
            melhor_a = max(ACOES, key=lambda x: Q[(s, x)])
            if a != melhor_a:
                regret_acum += 1
            regret_serie.append(regret_acum)

            N[(s, a)] += 1
            s2  = amostrar(s, a)
            r   = recomp(s2)
            mq2 = max(Q[(s2, x)] for x in ACOES)
            Q[(s, a)] += alpha * (r + gamma * mq2 - Q[(s, a)])
            s = s2; passo += 1

    return regret_serie[:n_passos], Q

reg_eps, Q_eps = treinar('eps_greedy', seed=7)
reg_fe,  Q_fe  = treinar('funcao_exp', seed=7)

# ── Impressão ─────────────────────────────────────────────────
print("=" * 55)
print("  Regret acumulado: ε-greedy vs f(u,n)")
print("=" * 55)
print(f"\n  Após {len(reg_eps)} passos:")
print(f"    ε-greedy  (ε=0.1): {reg_eps[-1]} ações subótimas")
print(f"    f(u,n)    (k=1.0): {reg_fe[-1]}  ações subótimas")

reducao = (reg_eps[-1] - reg_fe[-1]) / reg_eps[-1] * 100
print(f"\n  Redução de regret com f(u,n): {reducao:.1f}%")
print("\n  Razão: f(u,n) concentra a exploração onde a incerteza é maior")
print("  (estados pouco visitados). ε-greedy distribui uniformemente,")
print("  incluindo estados já bem conhecidos — desperdício de exploração.")

# ── Gráfico ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
ax.plot(reg_eps, 'b-', lw=1.5, alpha=0.9, label='ε-greedy (ε=0.1)')
ax.plot(reg_fe,  'r-', lw=1.5, alpha=0.9, label='Função de exploração (k=1.0)')
ax.set_xlabel('Passo de tempo')
ax.set_ylabel('Regret acumulado')
ax.set_title('Regret acumulado ao longo do treinamento')
ax.legend(); ax.grid(True, alpha=0.3)

# Taxa instantânea (janela deslizante)
ax2 = axes[1]
w = 200
eps_d = np.diff(reg_eps)
fe_d  = np.diff(reg_fe)
ma_eps = np.convolve(eps_d, np.ones(w)/w, mode='valid')
ma_fe  = np.convolve(fe_d,  np.ones(w)/w, mode='valid')
ax2.plot(ma_eps, 'b-', lw=1.5, label='ε-greedy')
ax2.plot(ma_fe,  'r-', lw=1.5, label='f(u,n)')
ax2.set_xlabel('Passo de tempo')
ax2.set_ylabel('Taxa de regret (média móvel)')
ax2.set_title('Taxa instantânea de ações subótimas')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.suptitle('Comparação de regret: ε-greedy vs função de exploração', fontsize=11)
plt.tight_layout()
plt.savefig('/home/claude/ar_exemplos/10_regret_acumulado.png',
            dpi=130, bbox_inches='tight')
plt.show()
