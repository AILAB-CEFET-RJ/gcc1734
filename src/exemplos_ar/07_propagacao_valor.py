"""
07_propagacao_valor.py — Seção 5: Propagação de valor
Cadeia linear determinística: s1 → s2 → s3 → s4 (terminal, r=+1).
Mostra que o valor se propaga a partir do terminal: s3 converge
primeiro, depois s2, depois s1 — exatamente como descrito no texto.
"""
import matplotlib.pyplot as plt
import numpy as np

CADEIA   = ['s1', 's2', 's3']
TERMINAL = 's4'
PROX     = {'s1': 's2', 's2': 's3', 's3': TERMINAL}
ALPHA    = 0.5
GAMMA    = 0.9

# Valores ótimos: Q*(sk) = γ^(distância ao terminal)
Q_STAR = {
    's3': GAMMA ** 1,   # 1 passo do terminal
    's2': GAMMA ** 2,   # 2 passos
    's1': GAMMA ** 3,   # 3 passos
}

def treinar(n_ep):
    Q = {s: 0.0 for s in CADEIA}
    hist = {s: [0.0] for s in CADEIA}
    for ep in range(n_ep):
        s = 's1'
        while s != TERMINAL:
            s2    = PROX[s]
            r     = 1.0 if s2 == TERMINAL else 0.0
            Q_s2  = Q.get(s2, 0.0)
            delta = r + GAMMA * Q_s2 - Q[s]
            Q[s] += ALPHA * delta
            s = s2
        for estado in CADEIA:
            hist[estado].append(Q[estado])
    return Q, hist

Q_final, hist = treinar(n_ep=60)

print("=" * 55)
print("  Propagação de valor — cadeia s1→s2→s3→s4(+1)")
print("=" * 55)
print(f"\n  α = {ALPHA}   γ = {GAMMA}\n")
print(f"  {'Estado':>7}  {'Q_final':>10}  {'Q*':>8}  {'Erro':>8}")
print("  " + "─" * 38)
for s in CADEIA:
    erro = abs(Q_final[s] - Q_STAR[s])
    print(f"  {s:>7}  {Q_final[s]:>10.4f}  {Q_STAR[s]:>8.4f}  {erro:>8.4f}")

print("\n  Observação:")
print("    Ep.  1: apenas Q(s3) recebe sinal positivo (vizinho do terminal)")
print("    Ep.  2: Q(s2) começa a subir (herda o valor de s3)")
print("    Ep. 3+: Q(s1) finalmente sobe (herda o valor de s2)")
print("    → Estados distantes do terminal precisam de mais episódios.")

# ── Gráfico ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
CORES = {'s1': '#e74c3c', 's2': '#e67e22', 's3': '#27ae60'}

for s in CADEIA:
    ax.plot(hist[s], color=CORES[s], lw=2.5, label=f'Q({s}, a)')
    ax.axhline(Q_STAR[s], color=CORES[s], ls='--', lw=1, alpha=0.7)
    # Anotação do valor ótimo
    ax.text(len(hist['s1']) + 0.3, Q_STAR[s],
            f' Q*({s})={Q_STAR[s]:.3f}',
            fontsize=8, color=CORES[s], va='center')

# Marca os episódios onde cada estado "acorda"
ep_acorda = {'s3': 1, 's2': 2, 's1': 3}
for s, ep in ep_acorda.items():
    ax.axvline(ep, color=CORES[s], ls=':', lw=1, alpha=0.6)
    ax.text(ep + 0.1, 0.03, f'ep.{ep}', fontsize=7, color=CORES[s])

ax.set_xlabel('Episódio', fontsize=11)
ax.set_ylabel('Q(s, a)', fontsize=11)
ax.set_title('Propagação de valor: convergência do terminal para o início', fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_xlim(-1, len(hist['s1']) + 6)
ax.set_ylim(-0.05, 1.1)

plt.tight_layout()
plt.savefig('/home/claude/ar_exemplos/07_propagacao_valor.png',
            dpi=130, bbox_inches='tight')
plt.show()
