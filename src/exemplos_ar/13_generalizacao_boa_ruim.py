"""
13_generalizacao_boa_ruim.py — Seção 7: Generalização indevida
Demonstra que dois estados com features semelhantes mas contextos
diferentes recebem valores Q parecidos quando usamos aproximação linear.
Mostra como uma feature adicional corrige o problema.
"""
import numpy as np
import matplotlib.pyplot as plt

T_POS = (4, 3)
T_NEG = (4, 2)
D_MAX = 6.0

# ── Features base (apenas 2: distâncias ao +1 e ao -1) ────────
def feat_base(s):
    d_pos = (abs(s[0]-T_POS[0]) + abs(s[1]-T_POS[1])) / D_MAX
    d_neg = (abs(s[0]-T_NEG[0]) + abs(s[1]-T_NEG[1])) / D_MAX
    return np.array([d_pos, d_neg])

def feat_com_adjacencia(s):
    """Feature adicional: está adjacente ao terminal negativo?"""
    adjacente = 1.0 if (abs(s[0]-T_NEG[0]) + abs(s[1]-T_NEG[1])) == 1 else 0.0
    return np.append(feat_base(s), adjacente)

ACOES = ['N', 'S', 'L', 'O']

# ── Par 1: generalização boa ─────────────────────────────────
s_a = (1, 1)   # canto inferior esquerdo
s_b = (2, 1)   # um passo à direita

# ── Par 2: generalização indevida ────────────────────────────
s_c = (3, 3)   # próximo do +1, ação Leste → T_POS  (ótima)
s_d = (3, 2)   # mesma coluna,  ação Leste → T_NEG  (péssima)

print("=" * 65)
print("  Generalização Boa vs Ruim — Aproximação Linear")
print("=" * 65)

print("\n  ── Par 1: generalização útil ────────────────────────────")
fa, fb = feat_base(s_a), feat_base(s_b)
print(f"  f({s_a})  = {np.round(fa, 4)}  →  Leste leva a {(s_a[0]+1, s_a[1])} (seguro)")
print(f"  f({s_b})  = {np.round(fb, 4)}  →  Leste leva a {(s_b[0]+1, s_b[1])} (seguro)")
print(f"  ‖Δf‖ = {np.linalg.norm(fa-fb):.4f}  — semelhantes E com consequências parecidas ✓")

print("\n  ── Par 2: generalização indevida ────────────────────────")
fc, fd = feat_base(s_c), feat_base(s_d)
print(f"  f({s_c})  = {np.round(fc, 4)}  →  Leste leva a {(s_c[0]+1, s_c[1])} = {T_POS}  🎯")
print(f"  f({s_d})  = {np.round(fd, 4)}  →  Leste leva a {(s_d[0]+1, s_d[1])} = {T_NEG}  ☠")
print(f"  ‖Δf‖ = {np.linalg.norm(fc-fd):.4f}  — semelhantes MAS consequências opostas ✗")

# ── Simulação com pesos simplificados ────────────────────────
print("\n  ── Q linear (2 features) para a ação Leste ─────────────")
# Pesos hipotéticos: o modelo aprendeu a valorizar "perto do objetivo"
N_F, N_A = 2, len(ACOES)
w = np.zeros(N_F * N_A)
idx_L = ACOES.index('L')
w[idx_L*N_F : (idx_L+1)*N_F] = np.array([-1.5, 0.3])  # penaliza dist_pos, ignora dist_neg

def phi(s, a, n_f=2):
    idx = ACOES.index(a)
    v   = np.zeros(n_f * N_A)
    v[idx*n_f : (idx+1)*n_f] = feat_base(s)
    return v

Q_sc = float(w @ phi(s_c, 'L'))
Q_sd = float(w @ phi(s_d, 'L'))
print(f"  Q({s_c}, Leste) = {Q_sc:+.4f}  (correto: deveria ser alto)")
print(f"  Q({s_d}, Leste) = {Q_sd:+.4f}  (errado: deveria ser baixo)")
print(f"  Diferença: {abs(Q_sc-Q_sd):.4f}  — modelo quase não os distingue!")

# ── Feature adicional que corrige o problema ──────────────────
print("\n  ── Com feature de adjacência ao terminal negativo ────────")
fc_ext = feat_com_adjacencia(s_c)
fd_ext = feat_com_adjacencia(s_d)
print(f"  f_ext({s_c}) = {np.round(fc_ext, 4)}  (não adjacente ao −1)")
print(f"  f_ext({s_d}) = {np.round(fd_ext, 4)}  (adjacente ao −1 ← flag ativo!)")
print(f"  ‖Δf_ext‖ = {np.linalg.norm(fc_ext-fd_ext):.4f}  — agora são distinguíveis ✓")
print("\n  Com um peso negativo forte em 'adjacente_neg', o modelo aprenderia")
print("  a penalizar (s_d, Leste), sem afetar (s_c, Leste).")

# ── Gráfico ───────────────────────────────────────────────────
ESTADOS_V = [(c,r) for c in range(1,5) for r in range(1,4) if (c,r) != (2,2)]
mat2 = np.array([feat_base(s) for s in ESTADOS_V])
mat3 = np.array([feat_com_adjacencia(s) for s in ESTADOS_V])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax_idx, (mat, nomes_col, titulo) in enumerate([
        (mat2, ['dist_obj','dist_neg'],         '2 features base'),
        (mat3, ['dist_obj','dist_neg','adj_neg'],'3 features (com adjacência)')]):
    ax = axes[ax_idx]
    im = ax.imshow(mat.T, aspect='auto', cmap='RdYlGn_r')
    ax.set_yticks(range(len(nomes_col))); ax.set_yticklabels(nomes_col, fontsize=8)
    ax.set_xticks(range(len(ESTADOS_V)))
    ax.set_xticklabels([str(s) for s in ESTADOS_V], rotation=45, fontsize=6)
    ax.set_title(titulo)
    plt.colorbar(im, ax=ax, shrink=0.8)
    for s, cor in [(s_c, 'blue'), (s_d, 'red')]:
        idx = ESTADOS_V.index(s)
        ax.axvline(idx, color=cor, lw=2.5, ls='--', alpha=0.8)

# Grade contextual
ax3 = axes[2]
ax3.set_xlim(0.5, 4.5); ax3.set_ylim(0.5, 3.5)
ax3.set_aspect('equal'); ax3.axis('off')
ax3.set_title('Contexto no Mundo Grade')
for col in range(1, 5):
    for row in range(1, 4):
        s = (col, row)
        if s == (2,2): continue
        fc = ('#90EE90' if s == T_POS else '#FFB6C1' if s == T_NEG else
              '#aed6f1' if s == s_c   else '#f9e79f' if s == s_d else 'white')
        ax3.add_patch(plt.Rectangle((col-.5,row-.5),1,1, ec='gray', fc=fc, lw=1))
        label = ('+1' if s==T_POS else '−1' if s==T_NEG else
                 f's_c\n→+1' if s==s_c else f's_d\n→−1' if s==s_d else '')
        color = ('black' if s in (T_POS,T_NEG) else 'blue' if s==s_c else
                 'red'   if s==s_d else 'black')
        if label:
            ax3.text(col, row, label, ha='center', va='center', fontsize=8, color=color)

ax3.text(2.5, 0.15, 'azul = s_c (seguro)   vermelho = s_d (perigoso)',
         ha='center', fontsize=7, style='italic')

plt.suptitle('Generalização boa vs indevida na aproximação linear', fontsize=11)
plt.tight_layout()
plt.savefig('/home/claude/ar_exemplos/13_generalizacao_boa_ruim.png',
            dpi=130, bbox_inches='tight')
plt.show()
