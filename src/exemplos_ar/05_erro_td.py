"""
05_erro_td.py — Seção 4: Erro de Diferença Temporal
Cadeia linear determinística: s1 → s2 → s3 → s4 (terminal, r=+1).
Compara MC (atualiza só ao fim do episódio) com TD(0) (atualiza
a cada passo). Mostra o cálculo do erro δ e plota a convergência.
"""
import numpy as np
import matplotlib.pyplot as plt

CADEIA   = ['s1', 's2', 's3']
TERMINAL = 's4'
PROX     = {'s1': 's2', 's2': 's3', 's3': TERMINAL}
ALPHA, GAMMA = 0.1, 1.0

def episodio():
    """Retorna a trajetória completa (sempre a mesma nesta cadeia)."""
    traj = ['s1']
    s = 's1'
    while s != TERMINAL:
        s = PROX[s]
        traj.append(s)
    return traj   # ['s1','s2','s3','s4']

# ── Exemplo numérico de um passo TD ──────────────────────────
print("=" * 55)
print("  Erro TD(0) — exemplo passo a passo")
print("=" * 55)
V_ex = {'s1': 0.3, 's2': 0.6, 's3': 0.9}
s_ex, s2_ex = 's2', 's3'
r_ex  = 0.0  # recompensa intermediária (só terminal tem r=+1)
delta = r_ex + GAMMA * V_ex[s2_ex] - V_ex[s_ex]
print(f"\n  Estado atual: {s_ex}  →  próximo: {s2_ex}")
print(f"  V({s_ex}) = {V_ex[s_ex]:.1f}   V({s2_ex}) = {V_ex[s2_ex]:.1f}   r = {r_ex:.1f}")
print(f"\n  δ = r + γ·V({s2_ex}) − V({s_ex})")
print(f"    = {r_ex:.1f} + {GAMMA}×{V_ex[s2_ex]:.1f} − {V_ex[s_ex]:.1f} = {delta:+.2f}")
print(f"\n  V({s_ex}) ← {V_ex[s_ex]:.1f} + α·δ = "
      f"{V_ex[s_ex]:.1f} + {ALPHA}×{delta:.2f} = "
      f"{V_ex[s_ex] + ALPHA*delta:.4f}")
sinal = "melhor" if delta > 0 else "pior" if delta < 0 else "igual"
print(f"\n  δ {'> 0' if delta > 0 else '< 0' if delta < 0 else '= 0'}: "
      f"transição foi {sinal} do que o esperado.")

# ── Treinamento: MC vs TD ─────────────────────────────────────
def treinar_mc(n_ep):
    V = {s: 0.0 for s in CADEIA}
    hist = {s: [0.0] for s in CADEIA}
    for _ in range(n_ep):
        traj = episodio()
        # MC: retorno real G calculado de trás para frente
        G = 0.0
        for i in range(len(traj) - 2, -1, -1):
            r = 1.0 if traj[i + 1] == TERMINAL else 0.0
            G = r + GAMMA * G
            s = traj[i]
            V[s] += ALPHA * (G - V[s])   # atualiza APÓS o episódio completo
        for s in CADEIA:
            hist[s].append(V[s])
    return hist

def treinar_td(n_ep):
    V = {s: 0.0 for s in CADEIA}
    hist = {s: [0.0] for s in CADEIA}
    for _ in range(n_ep):
        s = 's1'
        while s != TERMINAL:
            s2    = PROX[s]
            r     = 1.0 if s2 == TERMINAL else 0.0
            V_s2  = V.get(s2, 0.0)
            delta = r + GAMMA * V_s2 - V[s]
            V[s] += ALPHA * delta          # atualiza A CADA PASSO
            s = s2
        for s in CADEIA:
            hist[s].append(V[s])
    return hist

N_EP = 150
hist_mc = treinar_mc(N_EP)
hist_td = treinar_td(N_EP)

print(f"\n  Valores após {N_EP} episódios (V* = 1.0 para todos):")
print(f"  {'Estado':>6}  {'V_MC':>8}  {'V_TD':>8}  {'V*':>6}")
print("  " + "─" * 34)
for s in CADEIA:
    print(f"  {s:>6}  {hist_mc[s][-1]:>8.4f}  {hist_td[s][-1]:>8.4f}  {'1.0':>6}")

# ── Gráfico ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for i, s in enumerate(CADEIA):
    ax = axes[i]
    ax.plot(hist_mc[s], 'r-',  lw=1.5, alpha=0.85, label='Monte Carlo')
    ax.plot(hist_td[s], 'b-',  lw=1.5, alpha=0.85, label='TD(0)')
    ax.axhline(1.0, color='k', ls='--', lw=1, label='V* = 1.0')
    ax.set_title(f'Convergência de V({s})')
    ax.set_xlabel('Episódio'); ax.set_ylabel('V(s)')
    ax.set_ylim(-0.1, 1.25); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle('MC vs TD(0): V* = 1.0 para todos os estados da cadeia', fontsize=11)
plt.tight_layout()
plt.savefig('/home/claude/ar_exemplos/05_erro_td.png', dpi=130, bbox_inches='tight')
plt.show()
