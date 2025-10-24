# Linear Q-Learning Experiments

Este diretório contém scripts e classes para experimentos com **Q-Learning com aproximação linear** aplicados a ambientes do Gymnasium.  
O código foi desenvolvido para fins educacionais na disciplina **GCC1734 – Inteligência Artificial (CEFET/RJ)**.

---

## Estrutura dos arquivos

### `lql.py`
Implementa a classe **`QLearningAgentLinear`**, uma versão do Q-Learning em que a função de valor de ação é aproximada linearmente:

\[
Q(s, a) = \mathbf{w}^\top \, \phi(s, a)
\]

onde:
- \( \phi(s,a) \) é o vetor de *features* (extraído por `TaxiFeatureExtractor` ou `BlackjackFeatureExtractor`);
- \( \mathbf{w} \) são os pesos aprendidos via regressão incremental.

Principais recursos:
- Atualização baseada no erro temporal:  
  \( \Delta w = \alpha \, [r + \gamma \max_{a'} Q(s',a') - Q(s,a)] \, \phi(s,a) \)
- Normalização de *features* e inclusão de termo de *bias*;
- Controle de estabilidade numérica com *clipping* do erro temporal;
- Decaimento linear de ε (exploração);
- Inicialização de pesos centrada em zero;
- Suporte a ambientes genéricos com *feature extractors* customizados.

---

### `lql_train.py`
Script de treinamento e monitoramento do agente linear.

Funções principais:
- Inicializa o ambiente (`Taxi-v3` ou `Blackjack-v1`);
- Extrai *features* apropriadas via `taxi_feature_extractor` ou `blackjack_feature_extractor`;
- Treina o agente por um número definido de episódios;
- Armazena:
  - o agente treinado (`*-lql-agent.pkl`);
  - curvas de aprendizado (`*-learning_curve.png`);
  - decaimento de ε (`*-epsilons.png`);
  - gráfico resumo (`*-summary.png`).

Exemplo de execução:
```bash
python lql_train.py --env_name Taxi-v3 --num_episodes 8000 --plot
```

Argumentos principais:
| Parâmetro | Descrição | Padrão |
|------------|------------|---------|
| `--env_name` | Nome do ambiente Gymnasium (`Taxi-v3`, `Blackjack-v1`) | `Taxi-v3` |
| `--num_episodes` | Número de episódios de treinamento | `6000` |
| `--decay_rate` | Taxa de decaimento de ε | `0.0001` |
| `--learning_rate` | Taxa de aprendizado (α) | `0.05` |
| `--gamma` | Fator de desconto (γ) | `0.618` |
| `--plot` | Exibe os gráficos interativamente | — |

---

### `lql_play.py`
Script de execução e avaliação do agente linear já treinado.

Principais recursos:
- Carrega automaticamente o modelo salvo (`*-lql-agent.pkl`);
- Recria o ambiente com renderização automática:
  - `render_mode="human"` → janela gráfica (quando disponível);
  - `render_mode="ansi"` → saída textual (modo compatível com terminais);
- Permite execução interativa com `--render`;
- Exibe estatísticas médias de desempenho após múltiplos episódios.

Exemplo de uso:
```bash
python lql_play.py --env_name Taxi-v3 --num_episodes 5 --render
```

Argumentos principais:
| Parâmetro | Descrição | Padrão |
|------------|------------|---------|
| `--env_name` | Nome do ambiente Gymnasium | `Taxi-v3` |
| `--num_episodes` | Número de episódios de execução | `5` |
| `--max_steps` | Limite de passos por episódio | `500` |
| `--render` | Exibe o ambiente em tempo real | — |

Saída esperada:
```
Running agent on Taxi-v3 (ansi mode)...

Episode 1/5 — reward: 9, steps: 13
Episode 2/5 — reward: 8, steps: 15
...
******** Summary ********
Average episode length: 14.3
Average total reward: 8.6
*************************
```

---

### `taxi_feature_extractor.py` e `blackjack_feature_extractor.py`
Implementam extratores de *features* específicas para cada ambiente:
- `TaxiFeatureExtractor`: combina posição do táxi, destino e status do passageiro em um vetor binário;
- `BlackjackFeatureExtractor`: converte observações contínuas em representações numéricas normalizadas.

Esses extratores são responsáveis por mapear o espaço de estados contínuo ou simbólico em vetores adequados para o modelo linear.

---

## Saídas geradas
Durante o treinamento, são criados automaticamente:

| Arquivo | Conteúdo |
|----------|-----------|
| `*-lql-agent.pkl` | Agente Linear Q-Learning treinado |
| `*-learning_curve.png` | Curva de recompensa total por episódio |
| `*-epsilons.png` | Evolução do parâmetro ε |
| `*-summary.png` | Gráfico combinado de aprendizado |

---

## Resultados esperados

- **Taxi-v3**  
  O aprendizado é mais lento que na versão tabular, mas as recompensas médias convergem gradualmente para o intervalo **[+6, +9]** após alguns milhares de episódios.  
  Pequenas oscilações são esperadas devido à aproximação linear das *features* discretas.

- **Blackjack-v1**  
  A convergência é mais ruidosa e fortemente dependente da escolha das *features*.  
  Recompensas médias tendem a estabilizar próximas de zero, refletindo o equilíbrio natural do jogo.

---

## Dependências

- Python 3.10+  
- `gymnasium`  
- `numpy`  
- `matplotlib`  
- `pandas`  
- `scipy`

Instalação:
```bash
pip install gymnasium numpy matplotlib pandas scipy
```

---

## Licença
Uso livre para fins acadêmicos e de ensino.  
(C) CEFET/RJ – Escola de Informática e Computação.
