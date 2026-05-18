"""
02_mundo_grade_pdm.py — Seção 2: PDM do Mundo Grade 4×3
Define formalmente os cinco componentes do PDM: S, A, T, R, γ.
Verifica que Σ T(s,a,s') = 1 para todo (s,a).
Exibe T((1,1), N) em detalhe e simula um episódio estocástico.
"""
import random
from collections import defaultdict

# ── Componentes do PDM ───────────────────────────────────────
ESTADOS  = sorted({(c, r) for c in range(1, 5)
                   for r in range(1, 4)} - {(2, 2)})   # S
ACOES    = ['N', 'S', 'L', 'O']                        # A
GAMMA    = 0.9                                          # γ
T_POS, T_NEG = (4, 3), (4, 2)
DELTA    = {'N': (0, 1), 'S': (0, -1), 'L': (1, 0), 'O': (-1, 0)}
PERP     = {'N': ('O', 'L'), 'S': ('L', 'O'),          # perpendiculares
             'L': ('N', 'S'), 'O': ('S', 'N')}

def R(s):
    """Função de recompensa R(s) — depende apenas do estado destino."""
    return {T_POS: +1.0, T_NEG: -1.0}.get(s, -0.04)

def mover(s, a):
    c, r = s
    dc, dr = DELTA[a]
    s2 = (c + dc, r + dr)
    return s2 if s2 in ESTADOS else s

def T(s, a):
    """Modelo de transição: retorna dict {s': prob} para o par (s,a).
       0.8 → direção pretendida; 0.1 → cada perpendicular."""
    ae, ad = PERP[a]
    dist = defaultdict(float)
    dist[mover(s, a)]  += 0.8
    dist[mover(s, ae)] += 0.1
    dist[mover(s, ad)] += 0.1
    return dict(dist)

def amostrar(s, a):
    """Amostra um s' segundo T(s,a)."""
    d = T(s, a)
    return random.choices(list(d.keys()), weights=list(d.values()))[0]

# ── Informações gerais ────────────────────────────────────────
print("=" * 58)
print("  PDM — Mundo Grade 4×3")
print("=" * 58)
print(f"\n  |S| = {len(ESTADOS)} estados  |A| = {len(ACOES)} ações  γ = {GAMMA}\n")
print("  Componentes do PDM:")
print("    S  → 11 células não bloqueadas (inclui terminais)")
print("    A  → {N, S, L, O}")
print("    T  → estocástica: 0.8 pretendida, 0.1 cada perpendicular")
print("    R  → +1 terminal+, −1 terminal−, −0.04 demais")
print(f"    γ  → {GAMMA}\n")

# ── Verificação: Σ T(s,a,s') = 1 ────────────────────────────
erros = [(s, a) for s in ESTADOS for a in ACOES
         if abs(sum(T(s, a).values()) - 1.0) > 1e-9]
if erros:
    print(f"  ERRO: distribuições inválidas em {erros}")
else:
    print("  ✓  Σ T(s,a,s') = 1.00 para todo (s,a)\n")

# ── Exemplo detalhado: T((1,1), 'N') ─────────────────────────
print("  T((1,1), 'N')  — ação Norte a partir do canto inferior esquerdo:")
for s2, p in sorted(T((1,1), 'N').items(), key=lambda x: -x[1]):
    print(f"    P(s'={s2} | s=(1,1), a=N) = {p:.2f}   R(s')={R(s2):+.2f}")

# ── Episódio estocástico ──────────────────────────────────────
random.seed(7)
print("\n  Episódio estocástico a partir de (1,1)  [γ=0.9]:")
s, G, t = (1, 1), 0.0, 0
while s not in (T_POS, T_NEG) and t < 40:
    a  = random.choice(ACOES)
    s2 = amostrar(s, a)
    r  = R(s2)
    G += (GAMMA ** t) * r
    print(f"    t={t:>2}: {s} --{a}--> {s2}   r={r:+.2f}")
    s = s2; t += 1

print(f"\n  Retorno descontado do episódio: G = {G:.4f}")
