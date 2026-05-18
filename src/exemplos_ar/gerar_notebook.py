"""
gerar_notebook.py
Gera o notebook Jupyter unificado  notebook_AR.ipynb
a partir dos 14 scripts já implementados.
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

# ── Utilitários ───────────────────────────────────────────────
def md(src):
    return nbf.v4.new_markdown_cell(src)

def code(src):
    return nbf.v4.new_code_cell(src)

def code_from_file(path):
    """Lê um script .py e retorna uma célula de código sem o docstring."""
    txt = Path(path).read_text()
    return nbf.v4.new_code_cell(txt)

BASE = Path('/home/claude/ar_exemplos')

# ══════════════════════════════════════════════════════════════
# Célula de configuração global
# ══════════════════════════════════════════════════════════════
cells.append(md("""\
# Aprendizado por Reforço — Exemplos em Python
### GCC1734 Inteligência Artificial · CEFET/RJ

Este notebook reúne todos os exemplos de código que acompanham as
**Notas de Aula de Aprendizado por Reforço** (Prof. Eduardo Bezerra).
Cada seção corresponde a uma seção das notas e pode ser executada
de forma independente.

---
"""))

cells.append(code("""\
# ── Configuração global ───────────────────────────────────────
import random, numpy as np, matplotlib.pyplot as plt
from collections import defaultdict

%matplotlib inline
plt.rcParams.update({'figure.dpi': 110, 'font.size': 10})
random.seed(42); np.random.seed(42)

# ── Definição compartilhada do Mundo Grade 4×3 ───────────────
ESTADOS  = sorted({(c, r) for c in range(1, 5)
                   for r in range(1, 4)} - {(2, 2)})
NAO_T    = [s for s in ESTADOS if s not in ((4, 3), (4, 2))]
T_POS, T_NEG = (4, 3), (4, 2)
ACOES    = ['N', 'S', 'L', 'O']
DELTA    = {'N': (0,1), 'S': (0,-1), 'L': (1,0), 'O': (-1,0)}
PERP     = {'N': ('O','L'), 'S': ('L','O'), 'L': ('N','S'), 'O': ('S','N')}
D_MAX    = 6.0

def mover(s, a):
    c, r = s; dc, dr = DELTA[a]; s2 = (c+dc, r+dr)
    return s2 if s2 in ESTADOS else s

def amostrar(s, a):
    ae, ad = PERP[a]
    return random.choices([mover(s,a), mover(s,ae), mover(s,ad)],
                          weights=[0.8, 0.1, 0.1])[0]

def recomp(s):
    return {T_POS: +1.0, T_NEG: -1.0}.get(s, -0.04)

SETA = {'N':'↑','S':'↓','L':'→','O':'←'}

def plot_grade(politica, titulo='Política', ax=None):
    \"\"\"Plota a grade 4×3 com setas da política.\"\"\"
    standalone = ax is None
    if standalone: fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xlim(0.5, 4.5); ax.set_ylim(0.5, 3.5)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title(titulo)
    for col in range(1, 5):
        for row in range(1, 4):
            s = (col, row)
            fc = ('#444' if s==(2,2) else '#90EE90' if s==T_POS
                  else '#FFB6C1' if s==T_NEG else 'white')
            ax.add_patch(plt.Rectangle((col-.5,row-.5),1,1,ec='gray',fc=fc,lw=1.2))
            if   s == T_POS: ax.text(col,row,'+1',ha='center',va='center',
                                     fontsize=11,fontweight='bold')
            elif s == T_NEG: ax.text(col,row,'−1',ha='center',va='center',
                                     fontsize=11,fontweight='bold')
            elif s != (2,2) and s in politica:
                ax.text(col,row,SETA[politica[s]],ha='center',va='center',fontsize=22)
    if standalone: plt.tight_layout(); plt.show()

print("✓  Configuração carregada.")
"""))

# ══════════════════════════════════════════════════════════════
# Seção 1
# ══════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Seção 1 — Ciclo Agente–Ambiente
**Conceito:** vocabulário básico do AR — estado, ação, recompensa, transição.  
O agente age aleatoriamente. Nenhum aprendizado ocorre.
"""))
cells.append(code_from_file(BASE / '01_ciclo_ar.py'))

# ══════════════════════════════════════════════════════════════
# Seção 2
# ══════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Seção 2 — Processo de Decisão de Markov (PDM)
**Conceito:** os cinco componentes do PDM — S, A, T, R, γ.  
Verificação formal de que Σ T(s,a,s') = 1 e simulação estocástica.
"""))
cells.append(code_from_file(BASE / '02_mundo_grade_pdm.py'))

# ══════════════════════════════════════════════════════════════
# Seção 3a
# ══════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Seção 3a — Equação de Bellman (cálculo manual)
**Conceito:** Q*(s,a), V*(s) e política ótima π*(s).  
Reproduz **exatamente** o exemplo numérico do texto: Q((3,2), Norte) = 0.485.
"""))
cells.append(code_from_file(BASE / '03_bellman_manual.py'))

# ══════════════════════════════════════════════════════════════
# Seção 3b
# ══════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Seção 3b — Fator de Desconto γ
**Conceito:** agente paciente vs impaciente.  
Trajetórias A e B do texto; gráfico da utilidade em função de γ.
"""))
cells.append(code_from_file(BASE / '04_desconto.py'))

# ══════════════════════════════════════════════════════════════
# Seção 4
# ══════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Seção 4 — Aprendizado por Diferença Temporal (TD)
**Conceito:** erro TD δ, bootstrap, MC vs TD(0).  
Cadeia linear s1→s2→s3→s4 — mostra timing das atualizações.
"""))
cells.append(code_from_file(BASE / '05_erro_td.py'))

# ══════════════════════════════════════════════════════════════
# Seção 5a
# ══════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Seção 5a — Q-learning Tabular
**Conceito:** regra de atualização Q-learning, política ótima, curva de aprendizado.  
Mundo Grade 4×3 com transições estocásticas.
"""))
cells.append(code_from_file(BASE / '06_qlearning_tabular.py'))

# ══════════════════════════════════════════════════════════════
# Seção 5b
# ══════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Seção 5b — Propagação de Valor
**Conceito:** o valor se propaga do terminal para estados distantes ao longo dos episódios.  
Cadeia linear determinística — convergência estado a estado.
"""))
cells.append(code_from_file(BASE / '07_propagacao_valor.py'))

# ══════════════════════════════════════════════════════════════
# Seção 6a
# ══════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Seção 6a — SARSA vs Q-learning
**Conceito:** off-policy vs on-policy, trajetórias rente ao penhasco vs seguras.  
Ambiente Cliff Walking — o exemplo clássico de Sutton & Barto.
"""))
cells.append(code_from_file(BASE / '08_sarsa_vs_qlearning.py'))

# ══════════════════════════════════════════════════════════════
# Seção 6b
# ══════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Seção 6b — Estratégias de Exploração
**Conceito:** ε-greedy, softmax/Boltzmann e função de exploração f(u,n) = u + k/n.  
Tabelas de probabilidade e gráficos comparativos para o estado (1,1).
"""))
cells.append(code_from_file(BASE / '09_estrategias_exploracao.py'))

# ══════════════════════════════════════════════════════════════
# Seção 6c
# ══════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Seção 6c — Regret Acumulado
**Conceito:** quantidade de ações subótimas ao longo do treinamento.  
ε-greedy distribui exploração uniformemente; f(u,n) concentra onde há incerteza.
"""))
cells.append(code_from_file(BASE / '10_regret_acumulado.py'))

# ══════════════════════════════════════════════════════════════
# Seção 7a
# ══════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Seção 7a — Engenharia de Features
**Conceito:** features de estado, expansão em blocos para estado-ação,  
e como uma representação pobre provoca generalização indevida.
"""))
cells.append(code_from_file(BASE / '11_features_mundograde.py'))

# ══════════════════════════════════════════════════════════════
# Seção 7b
# ══════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Seção 7b — Q-learning com Aproximação Linear
**Conceito:** Q(s,a) = w · f(s,a), regra de atualização dos pesos,  
comparação com versão tabular.
"""))
cells.append(code_from_file(BASE / '12_qlearning_linear.py'))

# ══════════════════════════════════════════════════════════════
# Seção 7c
# ══════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Seção 7c — Generalização Boa vs Ruim
**Conceito:** dois estados com features semelhantes mas consequências opostas.  
Demonstra a necessidade de features que capturem risco local.
"""))
cells.append(code_from_file(BASE / '13_generalizacao_boa_ruim.py'))

# ══════════════════════════════════════════════════════════════
# Seção 8
# ══════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Seção 8 — Reward Shaping baseado em Potencial
**Conceito:** F(s,s') = γ·Φ(s') − Φ(s), shaping bom vs ruim,  
efeito na velocidade de convergência.
"""))
cells.append(code_from_file(BASE / '14_reward_shaping.py'))

# ══════════════════════════════════════════════════════════════
# Célula final
# ══════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Resumo dos Algoritmos

| Algoritmo | Tipo | Alvo de atualização | Generalização |
|-----------|------|----------------------|---------------|
| Q-learning tabular | off-policy | r + γ max Q(s',a') | Nenhuma |
| SARSA tabular | on-policy | r + γ Q(s',a') | Nenhuma |
| Q-learning linear | off-policy | r + γ max Q(s',a') | Via features |

> **Mensagem central:** o Q-learning tabular aprende valores exatos mas não escala.
> A aproximação linear escala e generaliza, mas exige boas features.
> A escolha da representação é tão importante quanto o algoritmo de aprendizado.
"""))

# ── Salva o notebook ──────────────────────────────────────────
nb.cells = cells
nb.metadata = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'name': 'python',
        'version': '3.10.0'
    }
}

OUT = BASE / 'notebook_AR.ipynb'
with open(OUT, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"✓  Notebook gerado: {OUT}")
print(f"   Células: {len(nb.cells)}")
