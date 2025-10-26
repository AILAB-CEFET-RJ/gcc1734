# Linear Function Approximation

Implementações e utilitários para **Q-Learning com aproximação linear** utilizados em sala na disciplina **GCC1734 – Inteligência Artificial (CEFET/RJ)**. Os agentes usam extratores de *features* específicos e compartilham o mesmo fluxo de treinamento da versão tabular.

---

## Estrutura

| Arquivo | Descrição |
| ------- | --------- |
| `qll.py` | Classe `QLearningAgentLinear` com política ε-greedy, atualização incremental sobre pesos `w` e *clipping* de erro temporal. |
| `qll_taxi_feature_extractor.py` / `qll_blackjack_feature_extractor.py` | Extratores de *features* responsáveis por transformar observações em vetores densos. |
| `train_qlearning.py` | Script genérico de treinamento para todas as variantes (tabular, linear, neural). |
| `play_qlearning.py` | Runner genérico; use `--agent linear` (ou `--agent neural`) para reproduzir políticas aproximadas. |

---

## Instalação recomendada

```bash
python -m venv .venv            # ou utilize conda
source .venv/bin/activate
pip install --no-build-isolation -e .
```

> Caso a instalação editável falhe por permissão, adicione `--user`.

Com o pacote instalado, os módulos podem ser referenciados como `rl.qll`, `rl.qll_taxi_feature_extractor`, etc., e o CLI unificado fica disponível via `python -m rl.train_qlearning`.

---

## Treinamento

> Execute os comandos a partir da raiz do repositório (`~/ailab/gcc1734`) com o ambiente ativado.

```bash
python -m rl.train_qlearning \
  --agent linear \
  --env_name Taxi-v3 \
  --num_episodes 8000 \
  --max_steps 500 \
  --plot
```

Parâmetros úteis (padrões em parênteses):

| Flag | Finalidade |
| ---- | ---------- |
| `--env_name {Taxi-v3,Blackjack-v1}` (`Taxi-v3`) | Ambiente Gymnasium escolhido. |
| `--num_episodes N` (`5000`) | Episódios de treinamento. |
| `--learning_rate LR` (`0.001`) | Taxa de atualização dos pesos. |
| `--gamma G` (`0.95`) | Fator de desconto. |
| `--decay_rate D` (`0.0005`) | Decaimento de ε para política ε-greedy. |
| `--max_steps` (`500`) | Limite de passos por episódio. |
| `--seed` (`42`) | Controle de aleatoriedade. |
| `--plot` | Abre os gráficos ao final do treinamento. |

## Artefatos produzidos

| Arquivo | Conteúdo |
| ------- | -------- |
| `taxi-v3-linear-agent.pkl` | Agente salvo via pickle. |
| `taxi-v3-linear-agent-learning_curve.png` | Recompensas por episódio (com suavização). |
| `taxi-v3-linear-agent-epsilons.png` | Histórico de ε. |
| `taxi-v3-linear-agent-summary.png` | Painel com recompensa × ε. |

Os nomes mudam conforme `env_name` e tipo de agente.

---

## Execução do agente

```bash
python -m rl.play_qlearning --agent linear --env_name Taxi-v3 --num_episodes 5
```

Funcionalidades:

- Carrega automaticamente `*-linear-agent.pkl` (ou utilize `--model_path`).
- Utiliza `render_mode="human"` quando o backend suporta, ou `render_mode="ansi"` (texto). Acrescente `--render` para exibir o ambiente.
- Interpreta ações com `policy(state)` (sem exploração) e gera estatísticas de recompensa média e passos por episódio.

---

## Expectativas de desempenho

- **Taxi-v3**: convergência mais lenta que a versão tabular, mas as recompensas tendem para valores positivos após milhares de episódios. Oscilações são comuns devido à aproximação linear.
- **Blackjack-v1**: evolução ruidosa, normalmente oscilando em torno de zero. O resultado depende fortemente das *features* escolhidas.

---

## Dependências principais

- Python 3.10+
- gymnasium
- numpy
- torch (apenas se usar o agente neural/MLP; ver documento correspondente)
- matplotlib
- scipy

Instalação manual (caso não utilize `pip install -e .`):

```bash
pip install gymnasium numpy matplotlib scipy
```

---

## Licença

Uso livre para fins acadêmicos e de ensino.  
(C) CEFET/RJ – Escola de Informática e Computação.
