# Tabular Q-Learning Experiments

Este diretório contém scripts e classes para experimentos didáticos com **Q-Learning tabular** usando ambientes do Gymnasium.  
O código foi projetado para fins educacionais na disciplina **GCC1734 – Inteligência Artificial (CEFET/RJ)**.

---

## Estrutura dos arquivos

### `tql.py`
Implementa a classe **`QLearningAgentTabular`**, um agente genérico para ambientes discretos.  
Principais recursos:
- Implementação direta da atualização Q-learning  
  \( Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)] \)
- Estratégia ε-greedy com decaimento exponencial.  
- Armazenamento da *Q-table* e histórico de recompensas e epsilons.  
- Métodos `save()` e `load_agent()` para persistência via pickle.  
- Código modular e com tipagem explícita.

---

### `tql_train.py`
Script de treinamento e visualização.  
Funções principais:
- Cria e inicializa o ambiente (`Taxi-v3` ou `Blackjack-v1`).  
- Treina o agente por um número definido de episódios.  
- Salva:
  - o agente treinado (`*.pkl`);
  - curvas de aprendizado (`*-learning_curve.png`);
  - decaimento de epsilon (`*-epsilons.png`);
  - gráfico resumo (`*-summary.png`).  
- Permite exibir os gráficos interativamente com `--plot`.

Exemplo de execução:
```bash
python tql_train.py --env_name Taxi-v3 --num_episodes 8000 --plot
```

Argumentos principais:
| Parâmetro | Descrição | Padrão |
|------------|------------|---------|
| `--env_name` | Nome do ambiente Gymnasium (`Taxi-v3` ou `Blackjack-v1`) | `Taxi-v3` |
| `--num_episodes` | Número de episódios de treinamento | `6000` |
| `--decay_rate` | Taxa de decaimento do ε | `0.0001` |
| `--learning_rate` | Taxa de aprendizado (α) | `0.7` |
| `--gamma` | Fator de desconto (γ) | `0.618` |
| `--seed` | Semente aleatória para reprodutibilidade | `42` |
| `--plot` | Exibe os gráficos interativamente | — |

---

### `tql_play.py`
Script de execução do agente treinado.  
Permite observar o comportamento da política aprendida em tempo real.

Principais recursos:
- Carrega automaticamente o agente salvo (`*-tql-agent.pkl`);  
- Recria o ambiente com o modo de renderização apropriado:  
  - `render_mode="human"` (janela gráfica, se suportado);  
  - `render_mode="ansi"` (saída textual no terminal, usada em `Taxi-v3` e ambientes de console);  
- Executa múltiplos episódios e exibe as trajetórias passo a passo;  
- Calcula e imprime estatísticas médias de recompensa e duração dos episódios.

Exemplo de uso:
```bash
python tql_play.py --env_name Taxi-v3 --num_episodes 5
```

Argumentos principais:
| Parâmetro | Descrição | Padrão |
|------------|------------|---------|
| `--env_name` | Nome do ambiente Gymnasium a ser carregado | `Taxi-v3` |
| `--num_episodes` | Número de episódios de execução | `5` |
| `--max_steps` | Limite máximo de passos por episódio | `500` |

Saída esperada:
```
Running agent on Taxi-v3 (ansi mode)...

Episode 1/5 finished — reward: 9, steps: 13
Episode 2/5 finished — reward: 8, steps: 15
...
******** Summary ********
Average episode length: 14.3
Average rewards: 8.6
*************************
```

Essa execução permite avaliar visualmente se o agente convergiu para uma política próxima do ótimo — por exemplo, no **Taxi-v3**, o táxi deve encontrar o passageiro e entregá-lo no destino em poucos movimentos, com penalizações mínimas.

---

### `taxi_environment.py` e `blackjack_environment.py`
Wrappers simples que adaptam os ambientes do Gymnasium ao formato esperado pelo agente:  
- Métodos obrigatórios: `get_num_states()`, `get_num_actions()`, `get_state_id(state)`, `step(action)` e `reset()`.  
- Cada wrapper pode incluir funções específicas de pré-processamento ou discretização.

---

## Saídas geradas
Após o treinamento, são criados arquivos no diretório atual:

| Arquivo | Conteúdo |
|----------|-----------|
| `*-tql-agent.pkl` | Agente Q-Learning treinado |
| `*-learning_curve.png` | Recompensa total por episódio (suavizada) |
| `*-epsilons.png` | Decaimento de ε |
| `*-summary.png` | Gráfico conjunto (recompensa + ε) |

---

## Resultados esperados

- **Taxi-v3**  
  O aprendizado começa com recompensas muito negativas (≈ -800) e melhora gradualmente até convergir em torno de **+8 a +9** por episódio após ~5000 iterações.  
  A curva de ε deve decair suavemente de 1.0 até ≈ 0.05, refletindo a transição da exploração para a exploração dirigida por política aprendida.  

- **Blackjack-v1**  
  A convergência é mais lenta e ruidosa, pois o ambiente é estocástico e contém estados terminais curtos.  
  Espera-se um padrão oscilante em torno de recompensa média próxima de zero, com variações dependentes da política aprendida.

Esses comportamentos ilustram bem o processo de equilíbrio entre **exploração e exploração**, um ponto central no aprendizado por reforço tabular.

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
