# Tabular Q-Learning

Implementação didática de Q-Learning tabular utilizada na disciplina **GCC1734 – Inteligência Artificial (CEFET/RJ)**. O código roda sobre o Gymnasium e já inclui utilitários para treinar, salvar e visualizar o desempenho do agente.

---

## Arquivos principais

| Arquivo | Descrição |
| ------- | --------- |
| `qlt.py` | Classe `QLearningAgentTabular` com atualização Q-learning, política ε-greedy com decaimento exponencial e histórico completo (`rewards`, `penalties`, `epsilons`, `steps`). |
| `environment_taxi.py` / `environment_blackjack.py` | Wrappers que adaptam os ambientes nativos do Gymnasium (`Taxi-v3`, `Blackjack-v1`) para a API esperada pelo agente (métodos `reset`, `step`, `get_num_states`, etc.). |
| `train_qlearning.py` | CLI unificado para treinar agentes tabulares, lineares e neurais. |
| `play_qlearning.py` | Runner genérico; use `--agent tabular` para executar políticas tabulares salvas. |

---

## Instalação rápida

```bash
python -m venv .venv            # ou use conda
source .venv/bin/activate
pip install --no-build-isolation -e .  # instala o pacote rl em modo editável
```

> Se o diretório do ambiente não for gravável, acrescente `--user`.

Após a instalação, os módulos podem ser importados como `rl.qlt`, `rl.environment_taxi`, etc., e é possível executar os CLIs com `python -m rl.train_qlearning`.

---

## Treinamento

> Execute os comandos a partir da raiz do repositório (`~/ailab/gcc1734`) com o ambiente ativado.

O training script unificado aceita diferentes variantes de agente. Para a versão tabular:

```bash
python -m rl.train_qlearning \
  --agent tabular \
  --env_name Taxi-v3 \
  --num_episodes 8000 \
  --plot
```

Flags úteis (padrões entre parênteses):

| Flag | Finalidade |
| ---- | ---------- |
| `--env_name {Taxi-v3,Blackjack-v1}` (`Taxi-v3`) | Ambiente Gymnasium. |
| `--num_episodes N` (`6000`) | Total de episódios de treinamento. |
| `--learning_rate LR` (`0.7`) | Fator α de atualização. |
| `--gamma G` (`0.618`) | Fator de desconto. |
| `--decay_rate D` (`0.0001`) | Taxa do decaimento exponencial de ε. |
| `--min_epsilon` (`0.01`) / `--max_epsilon` (`1.0`) | Limites de exploração. |
| `--seed` (`42`) | Reprodutibilidade. |
| `--plot` | Exibe gráficos ao fim do treinamento. |
| `--quiet` | Suprime logs periódicos do agente. |

## Artefatos gerados

Após o treinamento, são gravados no diretório atual:

| Arquivo | Conteúdo |
| ------- | -------- |
| `taxi-v3-tql-agent.pkl` | Agente treinado (pickle). |
| `taxi-v3-tql-learning_curve.png` | Série de recompensas por episódio com suavização de Savitzky-Golay. |
| `taxi-v3-tql-epsilons.png` | Decaimento de ε. |
| `taxi-v3-tql-summary.png` | Recompensas e ε lado a lado. |

Os prefixos mudam conforme o ambiente selecionado.

---

## Execução do agente

```bash
python -m rl.play_qlearning --agent tabular --env_name Taxi-v3 --num_episodes 5
```

Recursos:

- Carrega automaticamente `*-tql-agent.pkl` (ajuste `--model_path` se necessário).
- Recria o ambiente com `render_mode="human"` (GUI quando suportado) ou `render_mode="ansi"` (texto).
- Executa múltiplos episódios, imprime métricas agregadas; utilize `--render` para visualizar o ambiente.

---

## Comportamento esperado

- **Taxi-v3**: recompensas iniciais muito negativas (≈ −800) que convergem para valores positivos (~8) após alguns milhares de episódios. O gráfico de ε deve cair suavemente até cerca de 0.05.
- **Blackjack-v1**: recompensas oscilam em torno de zero porque o ambiente é altamente estocástico; o objetivo principal é observar a estabilização do ε e a redução de penalidades.

Esses experimentos ilustram o trade-off entre exploração e exploração no contexto tabular.

---

## Licença

Uso livre para fins acadêmicos e de ensino.  
(C) CEFET/RJ – Escola de Informática e Computação.
