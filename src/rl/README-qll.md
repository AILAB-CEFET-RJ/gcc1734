# Linear Function Approximation

Implementações e utilitários para **Q-Learning com aproximação linear** utilizados em sala na disciplina **GCC1734 – Inteligência Artificial (CEFET/RJ)**. Os agentes usam extratores de *features* específicos e compartilham o mesmo fluxo de treinamento da versão tabular.

---

## Estrutura

| Arquivo | Descrição |
| ------- | --------- |
| `qll.py` | Classe `QLearningAgentLinear` com política ε-greedy e atualização incremental dos pesos `w` pela regra do Q-learning aproximado. |
| `qll_taxi_feature_extractor.py` / `qll_blackjack_feature_extractor.py` | Extratores de *features* responsáveis por transformar observações em vetores densos. |
| `ql_train.py` | Script genérico de treinamento para todas as variantes (tabular, linear, neural). |
| `ql_play.py` | Runner genérico; use `--agent linear` (ou `--agent neural`) para reproduzir políticas aproximadas. |

---

## Instalação recomendada

```bash
python -m venv .venv            # ou utilize conda
source .venv/bin/activate
pip install --no-build-isolation -e .
```

> Caso a instalação editável falhe por permissão, adicione `--user`.

Com o pacote instalado, os módulos podem ser referenciados como `rl.qll`, `rl.qll_taxi_feature_extractor`, etc., e o CLI unificado fica disponível via `python -m rl.ql_train`.

---

## Treinamento

> Execute os comandos a partir da raiz do repositório (`~/ailab/gcc1734`) com o ambiente ativado.

```bash
python -m rl.ql_train \
  --agent linear \
  --env_name Taxi-v3 \
  --num_episodes 8000 \
  --max_steps 200 \
  --learning_rate 0.02 \
  --gamma 0.95 \
  --epsilon_decay_rate 0.0005 \
  --plot
```

Parâmetros úteis (padrões em parênteses):

| Flag | Finalidade |
| ---- | ---------- |
| `--env_name {Taxi-v3,Blackjack-v1}` (`Taxi-v3`) | Ambiente Gymnasium escolhido. |
| `--num_episodes N` (`6000`) | Episódios de treinamento. |
| `--learning_rate LR` (`0.02` para `--agent linear`) | Taxa de atualização dos pesos. |
| `--gamma G` (`0.95`) | Fator de desconto. |
| `--epsilon_decay_rate D` (`0.0005` para `--agent linear`) | Taxa de decaimento exponencial de ε na política ε-greedy. |
| `--max_steps` (`200` para `--agent linear`) | Limite de passos por episódio. |
| `--seed` (`42`) | Controle de aleatoriedade. |
| `--plot` | Abre os gráficos ao final do treinamento. |

## Regra de atualização

O agente implementa a hipótese linear:

```text
Q(s, a) = w · f(s, a)
```

e, para cada transição `(s, a, r, s')`, atualiza os pesos segundo:

```text
δ = r + γ max_a' Q(s', a') - Q(s, a)
w_i <- w_i + α δ f_i(s, a)
```

Quando a transição encerra o episódio (`terminated` ou `truncated`), o termo de bootstrap é zerado, isto é, `max_a' Q(s', a') = 0` no alvo TD.

Essa formulação é a mesma apresentada nas notas de aula para Q-learning com aproximação linear.

## Representação por features

O comportamento do agente depende diretamente do extrator `f(s, a)` usado em cada ambiente.

Para `Taxi-v3`, o extrator atual usa uma representação **esparsa** e mais estruturada, incluindo:

- célula exata do táxi no grid;
- localização atual do passageiro (ou indicador de que ele já está a bordo);
- destino final;
- fase atual da tarefa;
- depósito-alvo relevante na fase corrente;
- interação entre posição do táxi e alvo atual;
- indicadores de `pickup` correto, `drop` correto, `pickup` ilegal, `drop` ilegal e colisão com parede.

Essas *features* procuram tornar explícitas as duas fases centrais do problema e reduzir ambiguidades que prejudicavam a aproximação linear com descritores excessivamente genéricos.

Para `Blackjack-v1`, o extrator usa *features* simples e interpretáveis, como:

- soma atual do jogador;
- carta aberta do dealer;
- presença de ás utilizável;
- indicadores binários associados a mãos altas e a carta forte do dealer.

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
python -m rl.ql_play --agent linear --env_name Taxi-v3 --num_episodes 5
```

Funcionalidades:

- Carrega automaticamente `*-linear-agent.pkl` (ou utilize `--model_path`).
- Utiliza `render_mode="human"` quando o backend suporta, ou `render_mode="ansi"` (texto). Acrescente `--render` para exibir o ambiente.
- Interpreta ações com `policy(state)` (sem exploração) e gera estatísticas de recompensa média e passos por episódio.

---

## Expectativas de desempenho

- **Taxi-v3**: convergência mais lenta que a versão tabular, mas as recompensas tendem para valores positivos após milhares de episódios. Oscilações são comuns devido à aproximação linear.
- **Blackjack-v1**: evolução ruidosa, normalmente oscilando em torno de zero. O resultado depende fortemente das *features* escolhidas; o extrator atual usa atributos simples como soma do jogador, carta aberta do dealer e presença de ás utilizável.

## Observações didáticas

- A versão linear compartilha pesos entre muitos pares `(s, a)`, o que permite generalização, mas também pode induzir generalizações indevidas quando as *features* não capturam distinções importantes do ambiente.
- Em `Taxi-v3`, a representação combina codificação por ação com uma descrição esparsa do estado e do alvo atual, o que melhora bastante o desempenho do agente linear nesse domínio.
- Em `Blackjack-v1`, as *features* são simples e intencionalmente interpretáveis, mas a qualidade da política aprendida continua bastante sensível à escolha da representação.

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
