"""
01_ciclo_ar.py — Seção 1: Ciclo agente–ambiente
Ilustra o ciclo básico: o agente observa um estado, escolhe
uma ação aleatória e recebe recompensa + próximo estado.
Nenhum aprendizado ocorre — objetivo: fixar o vocabulário.
"""
import random

# ── Mundo Grade 4×3 ──────────────────────────────────────────
ESTADOS  = {(c, r) for c in range(1, 5) for r in range(1, 4)} - {(2, 2)}
T_POS    = (4, 3)   # terminal com recompensa +1
T_NEG    = (4, 2)   # terminal com recompensa -1
ACOES    = ['N', 'S', 'L', 'O']
DELTA    = {'N': (0, 1), 'S': (0, -1), 'L': (1, 0), 'O': (-1, 0)}

def mover(s, a):
    """Transição determinística (sem estocasticidade)."""
    c, r = s
    dc, dr = DELTA[a]
    s2 = (c + dc, r + dr)
    return s2 if s2 in ESTADOS else s   # bate na parede: fica no lugar

def recompensa(s):
    return {T_POS: +1.0, T_NEG: -1.0}.get(s, -0.04)

def is_terminal(s):
    return s in (T_POS, T_NEG)

# ── Episódio com política aleatória ──────────────────────────
random.seed(42)
s = (1, 1)

print("=" * 60)
print("  Ciclo Agente–Ambiente  |  política aleatória")
print("=" * 60)
print(f"  Estado inicial: {s}")
print()
print(f"  {'Passo':>5}  {'s':>6}  {'a':>3}  {'r':>6}  {'s_novo':>6}")
print("  " + "─" * 38)

G = 0.0
for passo in range(1, 25):
    a      = random.choice(ACOES)
    s_novo = mover(s, a)
    r      = recompensa(s_novo)
    G     += r
    flag   = "  ← TERMINAL" if is_terminal(s_novo) else ""
    print(f"  {passo:>5}  {str(s):>6}  {a:>3}  {r:>6.2f}  {str(s_novo):>6}{flag}")
    if is_terminal(s_novo):
        break
    s = s_novo

print()
print(f"  Retorno total não descontado: G = {G:.2f}")
print()
print("  Vocabulário fixado:")
print("    s       → estado observado antes da ação")
print("    a       → ação executada pelo agente")
print("    r       → recompensa imediata recebida")
print("    s_novo  → próximo estado (resultado da ação)")
print("    (s, a, r, s_novo) = uma transição / experiência")
