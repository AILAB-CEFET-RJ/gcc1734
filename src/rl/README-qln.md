# Neural Q-Learning

Este material descreve a versão com **rede neural (MLP) + Experience Replay** do agente de Q-Learning usado na disciplina **GCC1734 – Inteligência Artificial (CEFET/RJ)**. A implementação utiliza PyTorch para aproximar \(Q(s,a)\) e compartilha o mesmo fluxo de execução do restante do pacote `rl`.

---

## Componentes principais

| Arquivo | Papel |
| ------- | ----- |
| `qln.py` | Define `QLearningAgentReplay`: rede feed-forward (duas camadas ocultas, ReLU), *replay buffer* e política ε-greedy com decaimento exponencial. |
| `qll_taxi_feature_extractor.py` / `qll_blackjack_feature_extractor.py` | Reuso dos extratores de *features* para gerar vetores de entrada da MLP. |
| `ql_train.py` | CLI unificado para treinar agentes tabular, linear e neural (`--agent neural`). |

---

## Requisitos

- Python 3.10+
- gymnasium
- numpy
- torch
- matplotlib
- scipy

Instalação recomendada (modo editável):

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch gymnasium numpy matplotlib scipy
pip install --no-build-isolation -e .  # adicione --user se precisar
```

---

## Como treinar

> Execute os comandos a partir da raiz do repositório (`~/ailab/gcc1734`) com o ambiente ativado.

```bash
python -m rl.ql_train \
  --agent neural \
  --env_name Taxi-v3 \
  --num_episodes 8000 \
  --max_steps 400 \
	--learning_rate 0.002 \
	--epsilon_decay_rate 0.0001 \
  --plot
```

Parâmetros específicos da versão neural:

| Flag | Descrição | Padrão |
| ---- | --------- | ------ |
| `--hidden_dim` | Número de neurônios nas camadas ocultas da MLP | `64` |
| `--batch_size` | Tamanho do minibatch amostrado do replay buffer | `64` |
| `--max_steps` | Limite de passos por episódio (controla coleta de experiências) | `500` |
| `--epsilon_decay_rate` | Taxa de decaimento exponencial de ε | `0.0005` |
| `--learning_rate` | Taxa de aprendizado do otimizador Adam | `0.001` |
| `--train_every` | (definido em código: 4) número de passos entre atualizações da rede |
| `--seed` | Controle de reprodutibilidade | `42` |

## Artefatos gerados

- `taxi-v3-neural-agent.pkl` – checkpoint contendo pesos do modelo, otimizador e hiperparâmetros de ε.
- `taxi-v3-neural-agent-learning_curve.png` – curva de recompensa (suavizada via Savitzky-Golay).
- `taxi-v3-neural-agent-epsilons.png` – histórico de ε por episódio.
- `taxi-v3-neural-agent-summary.png` – painel com recompensa × ε.

Os nomes refletem o ambiente (`env_name`) utilizado.

---

## Avaliação

Utilize o runner genérico para reexecutar a política aprendida:

```bash
python -m rl.ql_play --agent neural --env_name Taxi-v3 --num_episodes 5 --max_steps 500
```

> Acrescente `--render` para visualizar o ambiente. Caso o backend não suporte renderização gráfica, será usada saída textual (`ansi`).

---

## Observações didáticas

- O replay buffer (deque com 50 000 transições) reduz a correlação entre amostras consecutivas e melhora a estabilidade do treinamento.
- O uso da perda Huber (`SmoothL1Loss`) protege contra outliers no TD error.
- Gradientes são *clipped* (`max_norm = 5.0`) para evitar explosão.
- O desempenho depende bastante da escolha de *features*: o extrator para Taxi funciona bem, enquanto Blackjack continua difícil por natureza.

---

## Licença

Uso livre para fins acadêmicos e de ensino.  
(C) CEFET/RJ – Escola de Informática e Computação.
