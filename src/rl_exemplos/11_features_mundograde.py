"""
11_features_mundograde.py — Seção 7: Engenharia de features
Define features para o Mundo Grade e demonstra:
  • Cálculo de f(s) para diferentes estados
  • Expansão de features de estado → features de estado-ação (blocos)
  • Por que features só de estado não bastam para distinguir Q(s,a)
"""
import numpy as np
import matplotlib.pyplot as plt

T_POS = (4, 3)
T_NEG = (4, 2)

# ── Definição das features de estado ─────────────────────────
def dist_manhattan(s, destino):
    return abs(s[0] - destino[0]) + abs(s[1] - destino[1])

D_MAX = 6.0   # máxima distância Manhattan no Mundo Grade 4×3

def f1_dist_obj(s):
    """Distância normalizada ao terminal positivo [0,1]."""
    return dist_manhattan(s, T_POS) / D_MAX

def f2_prox_perigo(s):
    """Proximidade do terminal negativo: 1/(1+d), mais alto = mais perto."""
    return 1.0 / (1 + dist_manhattan(s, T_NEG))

def f3_alinhado_col(s):
    """1 se mesma coluna que o objetivo."""
    return 1.0 if s[0] == T_POS[0] else 0.0

def f4_alinhado_lin(s):
    """1 se mesma linha que o objetivo."""
    return 1.0 if s[1] == T_POS[1] else 0.0

FEATURES     = [f1_dist_obj, f2_prox_perigo, f3_alinhado_col, f4_alinhado_lin]
NOMES_FEAT   = ['dist_obj', 'prox_perigo', 'alin_col', 'alin_lin']
ACOES        = ['N', 'S', 'L', 'O']
N_F          = len(FEATURES)

def vetor_estado(s):
    return np.array([f(s) for f in FEATURES])

def vetor_estado_acao(s, a):
    """
    Expande features de estado para features de estado-ação.
    Vetor de tamanho N_F × |A| com um bloco por ação.
    Apenas o bloco da ação escolhida é não-zero.
    """
    idx = ACOES.index(a)
    v   = np.zeros(N_F * len(ACOES))
    v[idx * N_F : (idx + 1) * N_F] = vetor_estado(s)
    return v

# ── Features de estado ────────────────────────────────────────
print("=" * 66)
print("  Features no Mundo Grade — Seção 7")
print("=" * 66)
print("\n  ── Features de ESTADO f(s) ──────────────────────────────")
print(f"  {'Estado':>8}  " + "  ".join(f"{n:>12}" for n in NOMES_FEAT))
print("  " + "─" * 62)
estados_demo = [(1,1), (2,1), (3,1), (4,1), (1,2), (3,2), (3,3), (4,3), (4,2)]
for s in estados_demo:
    vals = vetor_estado(s)
    print(f"  {str(s):>8}  " + "  ".join(f"{v:>12.4f}" for v in vals))

# ── Features de estado-ação ───────────────────────────────────
s_ex = (3, 2)
print(f"\n  ── Features de ESTADO-AÇÃO f(s,a)  para s={s_ex} ───────────")
print("  (apenas o bloco da ação selecionada é não-zero)")
nomes_bloco = [f"{a}_{n}" for a in ACOES for n in NOMES_FEAT]
print(f"\n  {'Ação':>4}   vetor f(s,a) [truncado p/ legibilidade]")
print("  " + "─" * 62)
for a in ACOES:
    v     = vetor_estado_acao(s_ex, a)
    bloco = v[ACOES.index(a)*N_F : (ACOES.index(a)+1)*N_F]
    zeros = np.zeros(N_F)
    blocos_str = ' | '.join(
        f"[{', '.join(f'{x:.3f}' for x in (bloco if i == ACOES.index(a) else zeros))}]"
        for i in range(len(ACOES)))
    print(f"  {a:>4}   {blocos_str}")

# ── Limite da representação: estados perigosamente similares ──
print("\n  ── Representação insuficiente: mesmas features, contextos opostos ─")
s_c = (3, 3)   # próximo do objetivo, ação Leste → terminal +1
s_d = (3, 2)   # mesma coluna,        ação Leste → terminal -1
print(f"\n  s_c = {s_c}:  f(s_c) = {np.round(vetor_estado(s_c), 4)}")
print(f"  s_d = {s_d}:  f(s_d) = {np.round(vetor_estado(s_d), 4)}")
dist = np.linalg.norm(vetor_estado(s_c) - vetor_estado(s_d))
print(f"  ‖f(s_c) − f(s_d)‖ = {dist:.4f}  (vetores parecidos)")
print(f"\n  Ação Leste de s_c → {(s_c[0]+1, s_c[1])} = {T_POS}  → Terminal +1 🎯")
print(f"  Ação Leste de s_d → {(s_d[0]+1, s_d[1])} = {T_NEG}  → Terminal −1 ☠")
print("\n  → Com esta representação, Q-linear atribuirá valores semelhantes")
print("    para (s_c, Leste) e (s_d, Leste). Feature extra necessária!")

# ── Gráfico ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Heatmap de features por estado
matrix = np.array([vetor_estado(s) for s in estados_demo])
ax = axes[0]
im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)
ax.set_xticks(range(N_F)); ax.set_xticklabels(NOMES_FEAT, rotation=20, fontsize=8)
ax.set_yticks(range(len(estados_demo)))
ax.set_yticklabels([str(s) for s in estados_demo], fontsize=8)
ax.set_title('Heatmap das features de estado f(s)')
plt.colorbar(im, ax=ax)
# Destaca os estados problemáticos
for s, cor in [(s_c, 'blue'), (s_d, 'red')]:
    idx = estados_demo.index(s)
    for j in range(N_F):
        ax.add_patch(plt.Rectangle((j-.5, idx-.5), 1, 1,
                                   fill=False, ec=cor, lw=2.5))

# Grade com contexto
ax2 = axes[1]
ax2.set_xlim(0.5, 4.5); ax2.set_ylim(0.5, 3.5)
ax2.set_aspect('equal'); ax2.axis('off')
ax2.set_title(f'Contexto: s_c e s_d têm features parecidas\nmas Leste leva a destinos opostos')
ESTADOS_ALL = {(c,r) for c in range(1,5) for r in range(1,4)} - {(2,2)}
for col in range(1, 5):
    for row in range(1, 4):
        s = (col, row)
        if s == (2,2):   continue
        fc = ('#444'    if s == (2,2)   else
              '#90EE90' if s == T_POS   else
              '#FFB6C1' if s == T_NEG   else
              '#aed6f1' if s == s_c     else
              '#f9e79f' if s == s_d     else 'white')
        ax2.add_patch(plt.Rectangle((col-.5, row-.5), 1, 1, ec='gray', fc=fc, lw=1))
        if   s == T_POS: ax2.text(col, row, '+1', ha='center', va='center', fontsize=10, fontweight='bold')
        elif s == T_NEG: ax2.text(col, row, '−1', ha='center', va='center', fontsize=10, fontweight='bold')
        elif s == s_c:   ax2.text(col, row, 's_c\n→+1', ha='center', va='center',
                                  fontsize=8, color='blue')
        elif s == s_d:   ax2.text(col, row, 's_d\n→−1', ha='center', va='center',
                                  fontsize=8, color='red')

plt.tight_layout()
plt.savefig('/home/claude/ar_exemplos/11_features_mundograde.png',
            dpi=130, bbox_inches='tight')
plt.show()
