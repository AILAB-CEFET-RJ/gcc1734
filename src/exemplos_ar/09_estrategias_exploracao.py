"""
09_estrategias_exploracao.py — Seção 6: Estratégias de exploração
Para o estado (1,1) com os Q-valores e contadores do texto,
compara três estratégias:
  (1) ε-greedy
  (2) Softmax/Boltzmann
  (3) Função de exploração  f(u, n) = u + k/n
"""
import numpy as np
import matplotlib.pyplot as plt

# ── Dados do exemplo do texto (estado (1,1)) ─────────────────
ACOES = ['N', 'S', 'L', 'O']
Q     = {'N':  0.41, 'S': -0.12, 'L':  0.28, 'O': -0.04}
N_vis = {'N':  50,   'S':  3,    'L':  12,    'O':  1}

# ── Estratégia 1: ε-greedy ────────────────────────────────────
def eps_greedy_probs(Q, eps):
    melhor = max(Q, key=Q.get)
    n = len(Q)
    return {a: (1 - eps + eps / n if a == melhor else eps / n) for a in Q}

# ── Estratégia 2: Softmax ─────────────────────────────────────
def softmax_probs(Q, tau):
    vals = np.array([Q[a] for a in ACOES])
    e    = np.exp(vals / tau)
    return {a: e[i] / e.sum() for i, a in enumerate(ACOES)}

# ── Estratégia 3: Função de exploração ───────────────────────
def funcao_exp(Q, N, k=1.0):
    f = {a: Q[a] + k / N[a] for a in ACOES}
    melhor = max(f, key=f.get)
    return {a: (1.0 if a == melhor else 0.0) for a in ACOES}, f

# ── Impressão das tabelas ─────────────────────────────────────
print("=" * 64)
print("  Estratégias de Exploração — estado (1,1)")
print("=" * 64)
print(f"\n  Q-valores:  N={Q['N']:.2f}  S={Q['S']:.2f}"
      f"  L={Q['L']:.2f}  O={Q['O']:.2f}")
print(f"  Visitas:    N={N_vis['N']:>3}   S={N_vis['S']:>3}"
      f"   L={N_vis['L']:>3}   O={N_vis['O']:>3}\n")

print("  ── ε-greedy ─────────────────────────────────────────")
print(f"  {'ε':>5}  {'P(N)':>8}  {'P(S)':>8}  {'P(L)':>8}  {'P(O)':>8}")
for eps in [0.0, 0.05, 0.10, 0.30, 1.00]:
    p = eps_greedy_probs(Q, eps)
    print(f"  {eps:>5.2f}  " + "  ".join(f"{p[a]:>8.4f}" for a in ACOES))

print()
print("  ── Softmax (Boltzmann) ──────────────────────────────")
print(f"  {'τ':>5}  {'P(N)':>8}  {'P(S)':>8}  {'P(L)':>8}  {'P(O)':>8}")
for tau in [0.01, 0.10, 0.30, 1.00, 5.00]:
    p = softmax_probs(Q, tau)
    print(f"  {tau:>5.2f}  " + "  ".join(f"{p[a]:>8.4f}" for a in ACOES))

print()
print("  ── Função de exploração  f(u,n) = u + k/n  (k=1.0) ──")
probs_fe, f_vals = funcao_exp(Q, N_vis, k=1.0)
print(f"  {'Ação':>5}  {'Q(s,a)':>8}  {'N(s,a)':>8}  {'f(Q,N)':>8}  Escolhida?")
for a in ACOES:
    flag = "  ← SIM" if probs_fe[a] == 1.0 else ""
    print(f"  {a:>5}  {Q[a]:>8.3f}  {N_vis[a]:>8}  {f_vals[a]:>8.3f}{flag}")

print("\n  → Oeste tem Q ruim, mas foi visitada apenas 1 vez:")
print("    o bônus de exploração eleva f(O) = −0.04 + 1.0/1 = 0.96")
print("    (acima de N com f(N) = 0.41 + 1.0/50 ≈ 0.43)")

# ── Gráfico ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
CORES = ['#2980b9', '#e74c3c', '#27ae60', '#f39c12']

# ε-greedy
ax = axes[0]
eps_range = [0.0, 0.05, 0.10, 0.20, 0.50, 1.00]
for a, c in zip(ACOES, CORES):
    ax.plot(eps_range, [eps_greedy_probs(Q, e)[a] for e in eps_range],
            'o-', color=c, lw=2, label=f'P({a})')
ax.set_xlabel('ε'); ax.set_ylabel('Probabilidade')
ax.set_title('ε-greedy'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Softmax
ax2 = axes[1]
taus = np.logspace(-2, 1.2, 80)
for a, c in zip(ACOES, CORES):
    ax2.semilogx(taus, [softmax_probs(Q, t)[a] for t in taus],
                 '-', color=c, lw=2, label=f'P({a})')
ax2.set_xlabel('τ (temperatura)'); ax2.set_ylabel('Probabilidade')
ax2.set_title('Softmax/Boltzmann'); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

# Bônus f(u,n) em função de N(s,a)
ax3 = axes[2]
ns = np.arange(1, 60)
for k, ls in zip([0.5, 1.0, 2.0], ['-', '--', ':']):
    ax3.plot(ns, Q['N'] + k / ns, lw=2, ls=ls, label=f'k={k}')
ax3.axhline(Q['N'], color='gray', lw=1, label=f'Q(N)={Q["N"]}')
ax3.set_xlabel('N(s,a) — nº de visitas')
ax3.set_ylabel('f(Q, N) = Q + k/N')
ax3.set_title('Bônus de exploração f(u,n)')
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3); ax3.set_ylim(-0.3, 2.5)

plt.suptitle('Comparação de estratégias de exploração', fontsize=11)
plt.tight_layout()
plt.savefig('/home/claude/ar_exemplos/09_estrategias_exploracao.png',
            dpi=130, bbox_inches='tight')
plt.show()
