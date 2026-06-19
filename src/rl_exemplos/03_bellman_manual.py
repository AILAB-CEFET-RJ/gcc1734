"""
03_bellman_manual.py — Seção 3: Equação de Bellman
Reproduz passo a passo os exemplos numéricos do texto:
  • Cálculo de Q((3,2), Norte) = 0.485
  • V*((3,2)) = max_a Q((3,2), a)
  • Política ótima em (3,2): Norte
"""

print("=" * 62)
print("  Equação de Bellman — exemplos do texto (Seção 3)")
print("=" * 62)

GAMMA   = 0.9
R_VIVO  = -0.04   # living reward para estados não terminais

# Valores V* fornecidos pelo texto para este exemplo
V = {(3, 3): 0.8,   # estado ao norte de (3,2)
     (3, 2): 0.5,   # o próprio estado (agente fica se bater no obstáculo)
     (4, 2): 0.0}   # terminal negativo: V* = 0 por definição

print("\n  Valores V*(s) assumidos:")
for estado, v in V.items():
    print(f"    V*{estado} = {v:.3f}")

# ── Q((3,2), Norte) ───────────────────────────────────────────
print("\n  ─── Cálculo de Q((3,2), Norte) ───\n")
print("  Ao executar Norte a partir de (3,2):")
print("    → (3,3) com prob 0.8  (direção pretendida)")
print("    → (3,2) com prob 0.1  (desvia p/ Oeste → bate no obstáculo em (2,2))")
print("    → (4,2) com prob 0.1  (desvia p/ Leste → Terminal −1)\n")

# Cada linha: (prob, s', recompensa imediata ao chegar em s')
transicoes = [
    (0.8, (3, 3), R_VIVO),   # chega em estado normal
    (0.1, (3, 2), R_VIVO),   # fica no lugar (bate no obstáculo)
    (0.1, (4, 2), -1.0),     # cai no terminal negativo
]

Q_norte = 0.0
for prob, s2, r in transicoes:
    v_futuro   = GAMMA * V[s2]
    parcela    = prob * (r + v_futuro)
    Q_norte   += parcela
    print(f"    {prob:.1f} × (r={r:+.2f} + γ·V*{s2})")
    print(f"    {prob:.1f} × ({r:+.2f} + {GAMMA}×{V[s2]:.3f})")
    print(f"    {prob:.1f} × {r + v_futuro:.4f} = {parcela:.4f}\n")

print(f"  Q((3,2), Norte) = {Q_norte:.4f}   [texto: 0.485]\n")

# ── V*((3,2)) e política ótima ───────────────────────────────
Q_leste = -0.2    # valor fornecido pelo texto (ação leva direto ao −1)

print("  ─── V*((3,2)) e política ótima ───\n")
print(f"  Q((3,2), Norte) = {Q_norte:.4f}  ← melhor ação")
print(f"  Q((3,2), Leste) = {Q_leste:.4f}  ← arriscada (Terminal −1 logo à direita)\n")

Q_vals   = {'Norte': Q_norte, 'Leste': Q_leste}
pi_star  = max(Q_vals, key=Q_vals.get)
V_star   = Q_vals[pi_star]

print(f"  V*((3,2)) = max_a Q((3,2), a) = {V_star:.4f}")
print(f"  π*((3,2)) = {pi_star}\n")
print("  → A política ótima prescrevê Norte: apesar da baixa probabilidade")
print("    de cair em (4,2), a ação Leste tem valor muito negativo porque")
print("    leva diretamente ao terminal −1 com alta probabilidade.")

# ── Tabela resumo ─────────────────────────────────────────────
print("\n  ─── Resumo da equação de Bellman ───\n")
print("    Q*(s, a) = Σ_{s'} T(s,a,s') · [R(s,a,s') + γ · V*(s')]")
print("    V*(s)    = max_a Q*(s, a)")
print("    π*(s)    = argmax_a Q*(s, a)")
