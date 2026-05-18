"""
04_desconto.py — Seção 3: Fator de desconto γ
Calcula a utilidade descontada das Trajetórias A e B do texto
para vários valores de γ e plota as curvas comparativas.

Trajetória A (curta):  r₀=0, r₁=0, r₂=+1
Trajetória B (longa):  r₀=0, r₁=0, r₂=0, r₃=0, r₄=+1
"""
import numpy as np
import matplotlib.pyplot as plt

def U(recompensas, gamma):
    """Utilidade descontada: U = Σ γ^t · r_t"""
    return sum(gamma ** t * r for t, r in enumerate(recompensas))

TRAJ_A = [0, 0, +1]         # recompensa em t=2
TRAJ_B = [0, 0, 0, 0, +1]   # recompensa em t=4

# ── Tabela de valores pontuais ────────────────────────────────
print("=" * 52)
print("  Fator de Desconto γ — Trajetórias A e B")
print("=" * 52)
print(f"\n  Trajetória A: {TRAJ_A}  (recompensa em t=2)")
print(f"  Trajetória B: {TRAJ_B}  (recompensa em t=4)\n")
print(f"  {'γ':>5}  {'U(A)':>8}  {'U(B)':>8}  {'A melhor?':>10}")
print("  " + "─" * 38)
for g in [0.10, 0.50, 0.90, 0.99]:
    ua = U(TRAJ_A, g)
    ub = U(TRAJ_B, g)
    print(f"  {g:>5.2f}  {ua:>8.4f}  {ub:>8.4f}  {'Sim' if ua > ub else 'Não':>10}")

print("\n  Interpretação:")
print("    γ baixo → agente impaciente → prefere fortemente A")
print("    γ alto  → agente paciente   → A e B ficam próximas")

# ── Gráfico ───────────────────────────────────────────────────
gammas = np.linspace(0.01, 0.99, 300)
UA = [U(TRAJ_A, g) for g in gammas]
UB = [U(TRAJ_B, g) for g in gammas]
diff = np.array(UA) - np.array(UB)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(gammas, UA, 'b-',  lw=2.5, label='Traj. A  (recomp. em t=2)')
ax.plot(gammas, UB, 'r--', lw=2.5, label='Traj. B  (recomp. em t=4)')
for g_mark in [0.5, 0.9]:
    ax.axvline(g_mark, color='gray', ls=':', lw=1)
    ax.text(g_mark + 0.01, 0.07, f'γ={g_mark}', fontsize=8, color='gray')
ax.set_xlabel('γ (fator de desconto)')
ax.set_ylabel('Utilidade descontada U')
ax.set_title('U(A) e U(B) em função de γ')
ax.legend(); ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(gammas, diff, 'g-', lw=2.5)
ax2.axhline(0, color='k', lw=0.8)
ax2.fill_between(gammas, diff, 0,
                 where=(diff > 0), alpha=0.25, color='blue',  label='A preferida')
ax2.fill_between(gammas, diff, 0,
                 where=(diff < 0), alpha=0.25, color='red',   label='B preferida')
ax2.set_xlabel('γ (fator de desconto)')
ax2.set_ylabel('U(A) − U(B)')
ax2.set_title('Diferença de utilidade A − B')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.suptitle('Efeito do fator de desconto γ sobre a preferência entre trajetórias',
             fontsize=11)
plt.tight_layout()
plt.savefig('/home/claude/ar_exemplos/04_desconto.png', dpi=130, bbox_inches='tight')
plt.show()
